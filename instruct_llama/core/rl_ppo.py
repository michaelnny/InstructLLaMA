# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Tuple, Optional, List, Mapping, Text, Any
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model import Transformer
from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.core.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.core.train_helper import (
    get_grad_norm_local,
    masked_whiten,
    masked_mean,
    masked_sum,
    masked_var,
    split_indices_into_bins,
    optimizer_to,
)
from instruct_llama.core.generation import top_k_logits, top_p_logits, sample_from_logits
from instruct_llama.core.custom_dataset import PromptOnlyDataset
from instruct_llama.utils.normalizer import Normalizer


def truncated_generalized_advantage_estimation(
    r_t: torch.Tensor,
    value_t: torch.Tensor,
    value_tp1: torch.Tensor,
    discount_tp1: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ + γₜ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Arguments:
      r_t: Sequence of rewards at times [0, k]
      value_t: Sequence of values under π at times [0, k]
      value_tp1: Sequence of values under π at times [1, k+1]
      discount_tp1: Sequence of discounts at times [1, k+1]
      lambda_: a scalar

    Returns:
      Multistep truncated generalized advantage estimation at times [0, k].
    """

    assert len(r_t.shape) == 1
    assert len(value_t.shape) == 1
    assert len(value_tp1.shape) == 1
    assert len(discount_tp1.shape) == 1

    lambda_ = torch.ones_like(discount_tp1) * lambda_  # If scalar, make into vector.

    delta_t = r_t + discount_tp1 * value_tp1 - value_t

    advantage_t = torch.zeros_like(delta_t, dtype=torch.float32)

    gae_t = 0
    for i in reversed(range(len(delta_t))):
        gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
        advantage_t[i] = gae_t

    return advantage_t


def find_begin_of_pattern(input_list: List[int], pattern: List[int] = [518, 25580, 29962]) -> int:
    """Find the beginning index of the some special token patterns from the given list"""
    assert len(pattern) > 1
    assert len(input_list) > len(pattern)

    pattern_length = len(pattern)
    lst_length = len(input_list)
    for i in range(lst_length - pattern_length + 1):
        if input_list[i : i + pattern_length] == pattern:
            return i

    return -1  # Return -1 if pattern is not found


def compute_logprobs_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3, logits.shape
    assert len(targets.shape) == 2, targets.shape
    assert logits.shape[:2] == targets.shape
    logprobs = torch.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, dim=2, index=targets.unsqueeze(2)).squeeze(2)


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3, logits.shape
    pd = torch.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def scale_data(data: torch.Tensor, target_min: float = -1.0, target_max: float = 1.0) -> torch.Tensor:
    """Scale data into range"""
    assert target_max > target_min
    max = torch.max(data)
    min = torch.min(data)
    if max == min:
        return data
    return (data - min) / (max - min) * (target_max - target_min) + target_min


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float = 0.02, adaptive: bool = False, target: float = 1.0, horizon: int = 10000) -> None:
        assert init_kl_coef >= 0
        assert target > 0
        assert horizon >= 100

        self.kl_coef = init_kl_coef
        self.adaptive = adaptive
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        if not self.adaptive:
            return

        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult

    @property
    def value(self) -> float:
        return self.kl_coef


class PPOAgent:
    """PPO agent for self-play and learning, using two separated networks for policy and value"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        policy_model: Transformer,
        policy_optimizer: torch.optim.AdamW,
        policy_scheduler: CosineDecayWithWarmupLRScheduler,
        value_model: Transformer,
        value_optimizer: torch.optim.AdamW,
        value_scheduler: CosineDecayWithWarmupLRScheduler,
        ref_model: Transformer,
        reward_model: Transformer,
        ppo_update_epochs: int,
        ppo_batch_size: int,
        policy_micro_bs: int,
        value_micro_bs: int,
        policy_clip_eps: float,
        value_clip_eps: float,
        entropy_coef: float,
        ptx_coef: float,
        discount: float,
        gae_lambda: float,
        policy_grad_clip: float,
        value_grad_clip: float,
        truncate_token: Optional[Tuple[str]],
        truncate_penalty_value: float,
        kl_ctl: AdaptiveKLController,
        scale_kl: bool,
        whiten_rewards: bool,
        whiten_advantages: bool,
        clip_reward: float,
        ref_device: str = 'cuda',
        reward_device: str = 'cuda',
        policy_device: str = 'cuda',
        value_device: str = 'cuda',
        ptx_loader: DataLoader = None,
        reward_normalizer_ckpt: str = None,
        tb_writer: SummaryWriter = None,
    ):
        """
        Arguments:
            tokenizer: tokenizer for the LLaMA model.
            policy_model: PPO policy model with LM head to be optimized.
            policy_optimizer: an torch optimizer for PPO policy model.
            policy_scheduler: an torch LR Scheduler for PPO policy model.
            value_model: PPO value model with scalar head to be optimized.
            value_optimizer: an torch optimizer for PPO value model.
            value_scheduler: an torch LR Scheduler for PPO value model.
            ref_model: trained reference policy model, frozen.
            reward_model: trained reward model, frozen.
            ppo_update_epochs: number of times to update the PPO models using rollout episodes.
            ppo_batch_size: batch size for PPO models during update.
            policy_micro_bs: micro-batch size for PPO policy model.
            value_micro_bs: micro-batch size for PPO value model.
            policy_clip_eps: PPO policy model clip epsilon.
            value_clip_eps: PPO value model clip epsilon.
            entropy_coef: coefficient for PPO policy model entropy loss.
            ptx_coef: coefficient for mix pre-training loss during PPO policy update.
            discount: PPO reward discount.
            gae_lambda: lambda for PPO GAE advantages.
            policy_grad_clip: gradient clip value for PPO policy model.
            value_grad_clip: gradient clip value for PPO value model.
            truncate_token: looking for first occurrence of special token text like `[INST]` in the response sequence.
            truncate_penalty_value: replace rewards with a fixed (negative) value as penalties to truncated tokens in the response.
            kl_ctl: an instance of the AdaptiveKLController.
            scale_kl: is true, remove negative KL and scale into [0, 1], so the coefficient always have the same meaning.
            whiten_rewards: if true, whiten rewards (combined with penalties) before using it to compute GAE advantages and returns.
            whiten_advantages: if true, whiten GAE advantages.
            clip_reward: if > 0, clip scalar reward (after normalize) in the range of [-clip_reward, clip_reward].
            ref_device: device for the fine-tuned model.
            reward_device: device for the reward model.
            policy_device: device for the PPO policy model.
            value_device: device for the PPO value model.
            ptx_loader: pre-training dataset loader, required if `ptx_coef > 0`.
            reward_normalizer_ckpt: checkpoint for the reward normalizer,
                if not provided, will compute running statistics for reward normalizer.
            tb_writer: tensorboard summary writer.
        """

        assert ppo_update_epochs >= 1, ppo_update_epochs
        assert policy_micro_bs >= 1, policy_micro_bs
        assert value_micro_bs >= 1, value_micro_bs
        assert ppo_batch_size > policy_micro_bs and ppo_batch_size > value_micro_bs, ppo_batch_size
        assert ppo_batch_size % policy_micro_bs == 0, policy_micro_bs
        assert ppo_batch_size % value_micro_bs == 0, value_micro_bs
        assert 0 <= policy_clip_eps < 0.5, policy_clip_eps
        assert 0 <= value_clip_eps < 0.5, value_clip_eps
        assert 0 <= entropy_coef < 0.1, entropy_coef
        assert ptx_coef >= 0, ptx_coef
        assert 0.9 <= discount <= 1, discount
        assert 0.9 <= gae_lambda <= 1, gae_lambda
        assert policy_grad_clip >= 0, policy_grad_clip
        assert value_grad_clip >= 0, value_grad_clip
        assert clip_reward >= 0, clip_reward
        if truncate_token is not None:
            assert isinstance(truncate_token, Tuple), truncate_token
            assert all(isinstance(t, str) for t in truncate_token), truncate_token
        assert truncate_penalty_value <= 0, truncate_penalty_value

        self.tokenizer = tokenizer
        self.ptx_loader = ptx_loader
        self.ref_device = ref_device
        self.reward_device = reward_device
        self.policy_device = policy_device
        self.value_device = value_device

        self.policy_model = policy_model.to('cpu')
        self.policy_optimizer = policy_optimizer
        self.policy_scheduler = policy_scheduler
        self.policy_grad_clip = policy_grad_clip
        self.model_params = self.policy_model.params

        self.value_model = value_model.to('cpu')
        self.value_optimizer = value_optimizer
        self.value_scheduler = value_scheduler
        self.value_grad_clip = value_grad_clip

        # reference policy and RM models are fixed, as we only need them to compute reward signals
        self.ref_model = ref_model.eval().to('cpu')
        self.reward_model = reward_model.eval().to('cpu')

        self.ppo_update_epochs = ppo_update_epochs
        self.ppo_batch_size = ppo_batch_size
        self.policy_micro_bs = policy_micro_bs
        self.value_micro_bs = value_micro_bs
        # gradient accumulation steps
        self.policy_accum_steps = self.ppo_batch_size // self.policy_micro_bs
        self.value_accum_steps = self.ppo_batch_size // self.value_micro_bs

        self.value_clip_eps = value_clip_eps
        self.policy_clip_eps = policy_clip_eps
        self.entropy_coef = entropy_coef
        self.ptx_coef = ptx_coef

        self.discount = discount
        self.gae_lambda = gae_lambda

        self.kl_ctl: AdaptiveKLController = kl_ctl
        self.scale_kl = scale_kl
        self.truncate_token = truncate_token
        self.truncate_penalty_value = truncate_penalty_value
        self.apply_truncate_penalty = True if self.truncate_token is not None and self.truncate_penalty_value < 0 else False

        self.whiten_rewards = whiten_rewards
        self.whiten_advantages = whiten_advantages
        self.clip_reward = clip_reward

        # normalize reward model output to have zero mean
        self.reward_normalizer = Normalizer(target_mean=0.0, target_std=1.0)
        self.update_reward_normalizer = True
        if reward_normalizer_ckpt is not None and os.path.exists(reward_normalizer_ckpt):
            norm_state = torch.load(reward_normalizer_ckpt)
            self.reward_normalizer.load_state_dict(norm_state)
            print(f'Reward normalizer state:\n{self.reward_normalizer.state_dict()}')
            self.update_reward_normalizer = False

        # store generated episodes for training
        self.buffer = []

        self.tb_writer = tb_writer
        # counters
        self.c_iteration = 0
        self.c_policy_update = 0
        self.c_value_update = 0
        self.c_train_episode = 0
        self.c_val_episode = 0

    def swap_model_device(self, name: str, to_cpu: bool = False):
        """Move model to device"""
        if name == 'policy':
            model = self.policy_model
            device = 'cpu' if to_cpu else self.policy_device
        elif name == 'value':
            model = self.value_model
            device = 'cpu' if to_cpu else self.value_device
        elif name == 'reward':
            model = self.reward_model
            device = 'cpu' if to_cpu else self.reward_device
        elif name == 'reference':
            model = self.ref_model
            device = 'cpu' if to_cpu else self.ref_device

        if model:
            model.to(device)
            torch.cuda.empty_cache()

    def swap_optimizer_device(self, name: str, to_cpu: bool = False):
        """Move optimizer to device"""
        if name == 'policy':
            device = 'cpu' if to_cpu else self.policy_device
            optimizer = self.policy_optimizer
        elif name == 'value':
            device = 'cpu' if to_cpu else self.value_device
            optimizer = self.value_optimizer

        if optimizer:
            optimizer_to(optimizer, device)

    @torch.no_grad()
    def run_selfplay(
        self,
        dataset: PromptOnlyDataset,
        num_episodes: int,
        batch_size: int,
        min_gen_len: int,
        max_gen_len: int,
        log_interval: int,
        sample_interval: int,
        temperature: float = 1.0,
        top_k: float = 0.0,
        top_p: float = 1.0,
        is_training: bool = False,
    ) -> None:
        """
        Arguments:
            dataset: a PromptOnlyDataset
            num_episodes: number of episodes to generate
            batch_size: batch size during selfplay generation
            min_gen_len: minimum number of tokens in the response
            max_gen_len: maximum number of tokens in the response
            log_interval: interval to log selfplay statistics, measured over number of episodes.
            sample_interval: interval to log generated text to tensorboard, measured in number of episodes.
            temperature: sampling temperature
            top_k: sampling top_k
            top_p: sampling top_p
            is_training: ito save computation we only compute certain values if is training.

        """

        self.policy_model.eval()
        self.value_model.eval()

        # poor solution to swap model device, so we can run on a single GPU
        if self.reward_device == self.policy_device:
            self.swap_model_device('reward', True)
        if self.ref_device == self.policy_device:
            self.swap_model_device('reference', True)
        if self.value_device == self.policy_device:
            self.swap_model_device('value', True)
            self.swap_optimizer_device('value', True)

        # make sure policy is on GPU for selfplay, moving model is not enough, we also need to move the optimizer
        self.swap_model_device('policy')
        self.swap_optimizer_device('policy', True)

        del self.buffer[:]

        batched_episodes = []

        # guarantee that we'll have at least M episodes
        episode_c = 0
        discard_c = 0
        while episode_c < num_episodes:
            episodes = self.generate_batch_episodes(dataset, batch_size, min_gen_len, max_gen_len, temperature, top_k, top_p)
            batched_episodes.append(episodes)
            current_bs = len(episodes['terminal_steps'])
            episode_c += current_bs
            discard_c += batch_size - current_bs

        if discard_c > 0:
            print(f'Discarded {discard_c} episodes with response tokens lesser than {min_gen_len}')

        if self.policy_device == self.reward_device or self.policy_device == self.value_device or self.policy_device == self.ref_device:
            self.swap_model_device('policy', True)

        # we do it in separate steps at the end of the epoch, so we can allocate one model at the time when using a single GPU
        self.swap_model_device('reward', False)
        self.compute_environment_reward_for_batched_episodes(batched_episodes)

        if self.reward_device == self.policy_device or self.reward_device == self.value_device or self.reward_device == self.ref_device:
            self.swap_model_device('reward', True)

        if is_training:
            self.swap_model_device('value', False)
            self.compute_state_values_for_batched_episodes(batched_episodes)
            if self.value_device == self.ref_device or self.value_device == self.policy_device:
                self.swap_model_device('value', True)

            self.swap_model_device('reference', False)
            self.compute_kl_for_batched_episodes(batched_episodes)
            if self.ref_device == self.policy_device or self.ref_device == self.value_device:
                self.swap_model_device('reference', True)

        episodes = self.flatten_batched_episodes(batched_episodes)
        self.log_selfplay_episode_stats(episodes, log_interval, sample_interval, is_training)

        if is_training:
            self.buffer = episodes

    @torch.no_grad()
    def generate_batch_episodes(
        self,
        dataset: PromptOnlyDataset,
        batch_size: int,
        min_gen_len: int,
        max_gen_len: int,
        temperature: float,
        top_k: float,
        top_p: float,
    ) -> Mapping[Text, torch.Tensor]:
        """Run one batch episodes, where the code is adapted from the `generation.py` module"""
        assert min_gen_len >= 1, min_gen_len
        assert min_gen_len < max_gen_len <= self.model_params.max_seq_len, max_gen_len
        assert batch_size >= 8, batch_size
        assert 0 < temperature <= 1.0, temperature
        assert 0 <= top_k <= 1.0, top_k
        assert 0 <= top_p <= 1.0, top_p

        self.policy_model.enable_cache()
        torch.cuda.empty_cache()

        # sample a batch of prompts token ids
        prompt_tokens = dataset.sample(batch_size)

        bsz = len(prompt_tokens)
        assert bsz <= self.model_params.max_batch_size, (bsz, self.model_params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        assert min_prompt_len > 6
        total_len = min(self.model_params.max_seq_len, max_prompt_len + max_gen_len)

        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.policy_device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.policy_device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.policy_device)
        input_text_mask = tokens != pad_id

        t0 = time.time()

        # RL agent starts selfplay
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.policy_model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits = logits[:, -1, :]  # [batch_size, vocab_size]

            if temperature > 0:
                logits = logits / temperature

            if top_k > 0 and top_k != 1.0:
                logits = top_k_logits(logits, top_k)

            if top_p > 0 and top_p != 1.0:
                logits = top_p_logits(logits, top_p)

            next_token = sample_from_logits(logits).reshape(-1)  # [batch_size]

            # only replace token if response has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        t1 = time.time()

        # start post-selfplay processing
        start_steps = torch.zeros((bsz,), dtype=torch.long, device=self.policy_device)
        terminal_steps = torch.zeros((bsz,), dtype=torch.long, device=self.policy_device)
        truncate_steps = torch.zeros((bsz,), dtype=torch.long, device=self.policy_device)

        # cut tokens
        for i, toks in enumerate(tokens.tolist()):
            start_idx = len(prompt_tokens[i])
            start_steps[i] = start_idx - 1  # starting from zero
            res_toks = toks[start_idx:]  # completion tokens

            # cut to eos token '</s>'
            if eos_id in res_toks:
                end_idx = start_idx + res_toks.index(eos_id)
                assert toks[end_idx] == eos_id
            else:
                # cut to max gen len
                end_idx = min(start_idx + max_gen_len - 1, len(toks) - 1)

            if end_idx > start_idx and end_idx < total_len:
                terminal_steps[i] = end_idx

            # looking for special tokens like '[INST]' and something alike, so we can later add a penalty value
            if self.apply_truncate_penalty:
                truncate_idx = 0
                for truncate_token in self.truncate_token:
                    truncate_token_ids = self.tokenizer.encode(truncate_token)
                    pen_start_idx = find_begin_of_pattern(res_toks, truncate_token_ids)
                    if pen_start_idx > 0:
                        truncate_idx = start_idx + pen_start_idx
                        if toks[truncate_idx : truncate_idx + len(truncate_token_ids)] != truncate_token_ids:
                            truncate_idx = 0
                        break

                if truncate_idx > 0 and truncate_idx > start_idx and truncate_idx < end_idx:
                    truncate_steps[i] = truncate_idx

        # filter episodes where length of response tokens are too short
        valid_episodes = ((terminal_steps - start_steps) >= min_gen_len).to(self.policy_device)
        num_discard = (~valid_episodes).sum().item()
        if num_discard > 0:
            tokens = tokens[valid_episodes, ...]
            start_steps = start_steps[valid_episodes]
            terminal_steps = terminal_steps[valid_episodes]
            truncate_steps = truncate_steps[valid_episodes]

        # build up loss mask, this mask is aligned with the target
        # for example, if we have a sequence of tokens:
        # [1, 2, 3, 4, 5, 6, 7, -1, -1]
        # where:
        #   [1, 2, 3, 4] are prompt tokens
        #   [5, 6, 7] are response tokens
        #   [-1, -1] are padding tokens
        #
        # then the mask will be:
        # [False, False, False, True, True, True, False, False, False]
        mask = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')

        # a mask for truncate tokens in the response sequence
        truncate_mask = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')
        for i, (start_t, end_t, truncate_t) in enumerate(zip(start_steps.tolist(), terminal_steps.tolist(), truncate_steps.tolist())):
            mask[i, start_t:end_t] = True
            if truncate_t > 0 and truncate_t < end_t:
                truncate_mask[i, truncate_idx:end_t] = True

        # replace pad ids
        tokens = torch.where(tokens == pad_id, eos_id, tokens)

        # shift one step to left to get the actions taken by the agent, this aligns with the RL transition convention: (s_t, a_t, logprob_a_t)
        actions = torch.full((len(tokens), total_len), eos_id, dtype=torch.long, device='cpu')
        actions[:, :-1] = tokens[:, 1:].clone().cpu()

        # compute action log probabilities, this is referred as the 'behavior policy' in RL terminology
        # apparently the log probabilities is slightly different when generating the tokens using attention cache
        # and with different temperature, top_k and top_p, which may introduce noise when we create KL
        # so we feed the tokens in one go and re-compute them here
        self.policy_model.disable_cache()
        torch.cuda.empty_cache()
        pi_logits = self.policy_model(tokens).cpu()
        logprobs = compute_logprobs_from_logits(pi_logits, actions)
        entropies = compute_entropy_from_logits(pi_logits)

        mean_time = torch.tensor((t1 - t0) / len(terminal_steps))
        episodes = {
            'tokens': tokens.cpu(),  # states s_t
            'actions': actions.cpu(),  # actions a_t
            'logprobs': logprobs.cpu(),  # log probabilities for actions logprob_a_t
            'mask': mask.cpu(),
            'truncate_mask': truncate_mask.cpu(),
            'entropies': entropies.cpu(),
            'start_steps': start_steps.cpu(),
            'terminal_steps': terminal_steps.cpu(),
            'truncate_steps': truncate_steps.cpu(),
            'episode_steps': (terminal_steps - start_steps).cpu(),
            'episode_time': mean_time.cpu(),
        }

        return episodes

    @torch.no_grad()
    def compute_state_values_for_batched_episodes(self, batched_episodes: List[Mapping[Text, torch.Tensor]]):

        for episodes in batched_episodes:
            tokens = episodes['tokens'].to(self.value_device)

            t0 = time.time()
            values = self.value_model(tokens).squeeze(-1).cpu()  # [batch_size, seq_len]

            episodes['values'] = values
            episodes['episode_time'] += (time.time() - t0) / len(tokens)

    @torch.no_grad()
    def compute_environment_reward_for_batched_episodes(self, batched_episodes: List[Mapping[Text, torch.Tensor]]):

        for episodes in batched_episodes:
            tokens = episodes['tokens']
            terminal_steps = episodes['terminal_steps']

            t0 = time.time()

            scalar_rewards = self.compute_environment_reward(tokens, terminal_steps).cpu()  # [batch_size]
            if self.update_reward_normalizer:
                self.reward_normalizer.update(scalar_rewards)
            scalar_rewards = self.reward_normalizer.normalize(scalar_rewards)

            if self.clip_reward > 0:
                scalar_rewards = torch.clamp(scalar_rewards, min=-self.clip_reward, max=self.clip_reward)

            # environment rewards are zero except for terminal step
            rewards = torch.zeros_like(tokens, dtype=torch.float, device='cpu')

            # shift one step to left because final reward is for state-action pair (s_T-1, a_T-1)
            # which is the time step prior reaching the EOS token
            # this also aligns with the rest of transition tuple (s_t, a_t, logprob_a_t, v_t, r_t)
            rewards[torch.arange(len(terminal_steps)), terminal_steps - 1] = scalar_rewards.clone()

            episodes['rewards'] = rewards.cpu()
            episodes['scalar_rewards'] = scalar_rewards.cpu()
            episodes['episode_time'] += (time.time() - t0) / len(tokens)

    @torch.no_grad()
    def compute_kl_for_batched_episodes(self, batched_episodes: List[Mapping[Text, torch.Tensor]]):

        for episodes in batched_episodes:
            tokens = episodes['tokens']
            actions = episodes['actions']
            logprobs = episodes['logprobs']
            mask = episodes['mask']

            t0 = time.time()
            ref_logprobs = self.compute_logprobs_from_reference_model(tokens, actions, logprobs).cpu()  # [batch_size, seqlen]
            kl = logprobs - ref_logprobs
            kl *= mask.float()
            episodes['kl'] = kl
            episodes['ref_logprobs'] = ref_logprobs
            episodes['episode_time'] += (time.time() - t0) / len(tokens)

    @torch.no_grad()
    def compute_environment_reward(self, tokens: torch.Tensor, terminal_steps: torch.Tensor) -> torch.Tensor:
        """Returns the environment rewards for the given batch of prompt-completion pairs"""

        assert len(tokens) == len(terminal_steps)
        assert len(tokens.shape) == 2
        assert len(terminal_steps.shape) == 1

        # compute the environment reward for the given completions
        tokens = tokens.to(self.reward_device)
        terminal_steps = terminal_steps.to(self.reward_device)
        outputs = self.reward_model(tokens).squeeze(-1)  # [batch_size, seq_length]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        env_rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [batch_size]
        return env_rewards

    @torch.no_grad()
    def compute_logprobs_from_reference_model(self, tokens: torch.Tensor, actions: torch.Tensor, logprobs: torch.Tensor) -> torch.Tensor:
        """Returns the log probabilities using the reference model for the given batch of prompt-completion pairs"""

        assert len(tokens) == len(logprobs) == len(actions)
        assert len(tokens.shape) == len(logprobs.shape) == len(actions.shape) == 2

        tokens = tokens.to(self.ref_device)
        actions = actions.to(self.ref_device)
        logprobs = logprobs.to(self.ref_device)

        ref_logits = self.ref_model(tokens)  # [batch_size, seq_len, vocab_size]
        ref_logprobs = compute_logprobs_from_logits(ref_logits, actions)
        return ref_logprobs

    def flatten_batched_episodes(self, batched_episodes: List[Mapping[Text, torch.Tensor]]) -> List[Mapping[Text, torch.Tensor]]:
        """Turn a list of batched episodes into a flat list of episodes"""
        results = []
        for episodes in batched_episodes:  # for each batch
            for i in range(len(episodes['terminal_steps'])):  # for each episode in current batch
                end_t = episodes['terminal_steps'][i]
                episode = {
                    'tokens': episodes['tokens'][i, :end_t],
                    'actions': episodes['actions'][i, :end_t],
                    'logprobs': episodes['logprobs'][i, :end_t],
                    'mask': episodes['mask'][i, :end_t],
                    'truncate_mask': episodes['truncate_mask'][i, :end_t],
                    'entropies': episodes['entropies'][i, :end_t],
                    'rewards': episodes['rewards'][i, :end_t],
                    'scalar_reward': episodes['scalar_rewards'][i],
                    'episode_steps': episodes['episode_steps'][i],
                    'episode_time': episodes['episode_time'],
                    'start_steps': episodes['start_steps'][i],
                    'terminal_steps': end_t,
                    'truncate_steps': episodes['truncate_steps'][i],
                }

                # only training episodes have these data
                for k in ('values', 'kl', 'ref_logprobs'):
                    if k in episodes:
                        episode[k] = episodes[k][i, :end_t]

                results.append(episode)

        return results

    @torch.no_grad()
    def compute_returns_and_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages for a single episode"""

        # here rewards, values are from t=0, 1, ..., T-2, T-1
        r_t = rewards
        v_t = values

        # pad value at t_p1 step to zero
        v_tp1 = torch.zeros_like(v_t, device='cpu')
        v_tp1[:-1] = v_t[1:]

        # mark terminal step
        done_tp1 = torch.zeros_like(v_tp1, dtype=torch.bool, device='cpu')
        done_tp1[-1:] = True

        discount_tp1 = (~done_tp1).float() * self.discount

        adv_t = truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, self.gae_lambda)
        return_t = adv_t + v_t
        return return_t, adv_t

    @torch.no_grad()
    def prepare_ppo_transitions(self) -> None:
        """Compute penalized rewards and GAE advantages for all episodes in current iteration"""
        batch_size = len(self.buffer)

        if batch_size < 8:
            raise ValueError(f'Please increase rollout size, current size is {batch_size}')

        max_seq_len = max([len(item['tokens']) for item in self.buffer])
        batched_kl = torch.full((batch_size, max_seq_len), 0.0, dtype=torch.float)
        batched_rewards = torch.full((batch_size, max_seq_len), 0.0, dtype=torch.float)
        batched_values = torch.full((batch_size, max_seq_len), 0.0, dtype=torch.float)
        batched_returns = torch.full((batch_size, max_seq_len), 0.0, dtype=torch.float)
        batched_advantages = torch.full((batch_size, max_seq_len), 0.0, dtype=torch.float)
        batched_mask = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)
        batched_truncate_mask = torch.full((batch_size, max_seq_len), 0, dtype=torch.bool)

        for i, item in enumerate(self.buffer):
            seq_len = len(item['tokens'])
            batched_kl[i, :seq_len] = item['kl']
            batched_rewards[i, :seq_len] = item['rewards']
            batched_values[i, :seq_len] = item['values']
            batched_mask[i, :seq_len] = item['mask']
            batched_truncate_mask[i, :seq_len] = item['truncate_mask']

        # Scale KL to [0, 1], so the coefficient always have the same meaning
        # as the raw KL values are very small, often in 9e-5 to 9e-7
        if self.scale_kl:
            # replace negative KLs with zero, as scaling may mess up the signs
            # this may also help prevent the agent try to 'gain' negative KL as reward signal, instead of the signal from the reward model???
            batched_kl = torch.where(batched_kl < 0, 0, batched_kl)
            batched_kl[batched_mask] = scale_data(batched_kl[batched_mask], target_min=0.0, target_max=1.0)

        # add pre-token KL penalties for the response tokens
        batched_kl_penalties = -self.kl_ctl.value * batched_kl
        batched_kl_penalties *= batched_mask.float()
        batched_rewards += batched_kl_penalties

        # replace rewards with a fixed (negative) value as penalties to truncated tokens in the response
        if self.apply_truncate_penalty:
            batched_rewards = torch.where(batched_truncate_mask, self.truncate_penalty_value, batched_rewards)

        if self.whiten_rewards:
            batched_rewards = masked_whiten(batched_rewards, batched_mask, shift_mean=False)

        # compute GAE advantages one episode at a time
        for i, item in enumerate(self.buffer):
            start_t = item['start_steps']
            end_t = item['terminal_steps']
            # here rewards, values with slice `start_t:end_t` are from t=0, 1, ..., T-2, T-1
            returns, advantages = self.compute_returns_and_advantages(batched_values[i, start_t:end_t], batched_rewards[i, start_t:end_t])
            batched_returns[i, start_t:end_t] = returns
            batched_advantages[i, start_t:end_t] = advantages

        if self.whiten_advantages:
            batched_advantages = masked_whiten(batched_advantages, batched_mask, shift_mean=True)

        # now add those returns and advantages back to the items in the buffer
        for i, item in enumerate(self.buffer):
            end_t = item['terminal_steps']
            item['returns'] = batched_returns[i, :end_t]
            item['advantages'] = batched_advantages[i, :end_t]
            item['kl_penalties'] = batched_kl_penalties[i, :end_t]
            item['truncate_penalties'] = batched_truncate_mask[i, :end_t].float() * self.truncate_penalty_value

    @torch.no_grad()
    def get_batched_transitions(self, batch: List[Mapping[Text, torch.Tensor]]) -> Mapping[Text, torch.Tensor]:
        """Essentially the same as a regular custom collate function for dataloader"""
        batch_size = len(batch)
        max_seq_len = max([len(item['tokens']) for item in batch])

        eos_id = self.tokenizer.eos_id

        batched_tokens = torch.full((batch_size, max_seq_len), eos_id, dtype=torch.long)
        batched_actions = torch.full((batch_size, max_seq_len), eos_id, dtype=torch.long)
        batched_logprobs = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_values = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_returns = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_advantages = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

        # the following are for logging
        batched_episode_steps = torch.zeros((batch_size,), dtype=torch.float)
        batched_episode_times = torch.zeros((batch_size,), dtype=torch.float)
        batched_scalar_rewards = torch.zeros((batch_size,), dtype=torch.float)
        batched_rewards = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_entropies = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_kl = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_kl_penalties = torch.zeros((batch_size, max_seq_len), dtype=torch.float)
        batched_truncate_penalties = torch.zeros((batch_size, max_seq_len), dtype=torch.float)

        for i, item in enumerate(batch):
            seq_len = len(item['tokens'])
            batched_tokens[i, :seq_len] = item['tokens'].clone()
            batched_actions[i, :seq_len] = item['actions'].clone()
            batched_logprobs[i, :seq_len] = item['logprobs'].clone()
            batched_values[i, :seq_len] = item['values'].clone()
            batched_returns[i, :seq_len] = item['returns'].clone()
            batched_advantages[i, :seq_len] = item['advantages'].clone()
            batched_mask[i, :seq_len] = item['mask'].clone()

            batched_episode_steps[i] = item['episode_steps'].clone()
            batched_episode_times[i] = item['episode_time'].clone()
            batched_scalar_rewards[i] = item['scalar_reward'].clone()
            batched_rewards[i, :seq_len] = item['rewards'].clone()
            batched_entropies[i, :seq_len] = item['entropies'].clone()
            batched_kl[i, :seq_len] = item['kl'].clone()
            batched_kl_penalties[i, :seq_len] = item['kl_penalties'].clone()
            batched_truncate_penalties[i, :seq_len] = item['truncate_penalties'].clone()

        return {
            'tokens': batched_tokens,
            'actions': batched_actions,
            'logprobs': batched_logprobs,
            'values': batched_values,
            'returns': batched_returns,
            'advantages': batched_advantages,
            'mask': batched_mask,
            # for logging
            'episode_steps': batched_episode_steps,
            'episode_time': batched_episode_times,
            'scalar_reward': batched_scalar_rewards,
            'rewards': batched_rewards,
            'entropies': batched_entropies,
            'kl': batched_kl,
            'kl_penalties': batched_kl_penalties,
            'truncate_penalties': batched_truncate_penalties,
        }

    def run_train(self, log_interval: int = 1) -> None:
        """
        Uses PPO and the RL agent generated selfplay episodes to train the policy network.

        Arguments:
            iteration: current iteration
            log_interval: interval to log training statistics, measured in number of model updates.
        """

        assert len(self.buffer) > 0

        t0 = time.time()

        # make room for policy model
        if self.reward_device == self.policy_device or self.reward_device == self.value_device:
            self.swap_model_device('reward', True)
        if self.ref_device == self.policy_device or self.ref_device == self.value_device:
            self.swap_model_device('reference', True)

        if self.value_device == self.policy_device:
            self.swap_model_device('value', True)
            self.swap_optimizer_device('value', True)

        # Compute GAE advantages once before starts training
        self.prepare_ppo_transitions()

        # Run M epochs to update policy model
        # we use two loops because we can't host the two models at the same time with a single GPU
        self.policy_model.disable_cache()
        self.swap_model_device('policy', False)
        self.swap_optimizer_device('policy', False)
        self.policy_model.train()
        torch.cuda.empty_cache()

        def get_micro_batches(size: int) -> List[Mapping[Text, torch.Tensor]]:
            # Split episodes into micro batches
            batch_indices = split_indices_into_bins(size, len(self.buffer), shuffle=True, drop_last=True)
            results = [self.get_batched_transitions([self.buffer[i] for i in indices]) for indices in batch_indices]
            return results

        for _ in range(self.ppo_update_epochs):
            # Dynamic batching can speed up training by ~20%
            micro_batches = get_micro_batches(self.policy_micro_bs)
            for j in range(0, len(micro_batches), self.policy_accum_steps):
                stats = self.update_policy_model(micro_batches[j : j + self.policy_accum_steps])
                if self.c_policy_update % log_interval == 0:
                    self.log_stats_to_tensorboard(stats, self.c_policy_update, 'ppo_policy')

        # make room for value model, we need the value model to compute state values during PPO transition preparation
        if self.policy_device == self.value_device:
            self.swap_model_device('policy', True)
            self.swap_optimizer_device('policy', True)

        self.value_model.disable_cache()
        self.swap_model_device('value', False)
        self.swap_optimizer_device('value', False)
        self.value_model.train()
        torch.cuda.empty_cache()

        # Run M epochs to update value model
        for _ in range(self.ppo_update_epochs):
            micro_batches = get_micro_batches(self.value_micro_bs)
            for j in range(0, len(micro_batches), self.value_accum_steps):
                stats = self.update_value_model(micro_batches[j : j + self.value_accum_steps])
                if self.c_value_update % log_interval == 0:
                    self.log_stats_to_tensorboard(stats, self.c_value_update, 'ppo_value')

        if self.value_device == self.policy_device:
            self.swap_model_device('value', True)
            self.swap_optimizer_device('value', True)

        t1 = time.time()
        self.c_iteration += 1

        # log aggregated statistics at the end of iteration
        iter_stats = self.compute_iteration_stats_from_episodes(self.buffer)
        iter_stats['objective/kl_coef'] = self.kl_ctl.value
        iter_stats['elapsed/policy_updates'] = self.c_policy_update
        iter_stats['elapsed/value_updates'] = self.c_value_update
        iter_stats['elapsed/episodes'] = self.c_train_episode
        iter_stats['elapsed/time'] += t1 - t0

        # split into separate tags for better readabilities
        def split_and_flatten_stats(stats_dict: Mapping[Text, Any], split_kw: str) -> Mapping[Text, Any]:
            assert split_kw is not None
            return {k.split('/')[1]: v for k, v in stats_dict.items() if split_kw in k and '/' in k}

        self.log_stats_to_tensorboard(split_and_flatten_stats(iter_stats, 'objective'), self.c_iteration, 'iterations_objective')
        self.log_stats_to_tensorboard(split_and_flatten_stats(iter_stats, 'elapsed'), self.c_iteration, 'iterations_elapsed')

        # Update adaptive KL controller
        self.kl_ctl.update(iter_stats['objective/kl'], len(self.buffer))

        del self.buffer[:]

    def compute_ptx_pretrain_loss(self, max_steps: int) -> List[float]:
        losses = []
        if self.ptx_loader is None:
            return losses

        assert max_steps > 1

        for i, (x, y) in enumerate(self.ptx_loader):
            x = x.to(self.policy_device)
            y = y.to(self.policy_device)

            y_pred = self.policy_model(x)
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), reduction='mean')

            loss *= self.ptx_coef
            scaled_loss = loss / max_steps
            scaled_loss.backward()

            losses.append(loss.detach().item())
            if i >= max_steps:
                break

        return losses

    def update_policy_model(
        self,
        micro_batches: List[Mapping[Text, torch.Tensor]],
    ) -> Mapping[Text, Any]:
        torch.cuda.empty_cache()
        self.policy_optimizer.zero_grad()

        stats = {
            'pg_loss': [],
            'entropy': [],
            'entropy_loss': [],
            'ptx_loss': [],
            'approxkl': [],
            'clipfrac': [],
        }

        t0 = time.time()
        # accumulate gradients over N micro batches
        for batch in micro_batches:
            tokens = batch['tokens'].to(dtype=torch.long, device=self.policy_device)
            actions = batch['actions'].to(dtype=torch.long, device=self.policy_device)
            behavior_logprobs = batch['logprobs'].to(dtype=torch.float, device=self.policy_device)
            advantages = batch['advantages'].to(dtype=torch.float, device=self.policy_device)
            mask = batch['mask'].to(dtype=torch.bool, device=self.policy_device)

            # Given past states, get predicted action probabilities
            pi_logits = self.policy_model(tokens)
            pi_logprobs = compute_logprobs_from_logits(pi_logits, actions)
            entropies = compute_entropy_from_logits(pi_logits)
            assert pi_logprobs.shape == behavior_logprobs.shape  # [batch_size, seqlen]

            # Compute PPO clipped surrogate objective
            ratio = torch.exp(pi_logprobs - behavior_logprobs)
            clipped_ratio = torch.clamp(ratio, min=1.0 - self.policy_clip_eps, max=1.0 + self.policy_clip_eps)
            pg_losses1 = ratio * advantages.detach()
            pg_losses2 = clipped_ratio * advantages.detach()
            pg_losses = torch.min(pg_losses1, pg_losses2)

            # apply loss mask
            assert entropies.shape == pg_losses.shape == mask.shape
            pg_loss = masked_mean(pg_losses, mask)
            entropy = masked_mean(entropies, mask)
            entropy_loss = self.entropy_coef * entropy

            # Negative sign to indicate we want to maximize the policy gradient objective function and entropy
            loss = -(pg_loss + entropy_loss)
            scaled_loss = loss / len(micro_batches)  # scale the loss to account for gradient accumulation
            scaled_loss.backward()

            approxkl = 0.5 * masked_mean(torch.square(pi_logprobs - behavior_logprobs), mask)
            clipfrac = masked_mean(torch.lt(pg_losses2, pg_losses1), mask)

            stats['pg_loss'].append(pg_loss.detach().item())
            stats['entropy'].append(entropy.detach().item())
            stats['approxkl'].append(approxkl.detach().item())
            stats['clipfrac'].append(clipfrac.detach().item())
            if self.entropy_coef > 0:
                stats['entropy_loss'].append(entropy_loss.detach().item())

        # Compute pre-training loss, mixing pre-training gradients into PPO policy
        # we do it separately because a single backward() call will cause CUDA OOM
        if self.ptx_coef > 0:
            ptx_losses = self.compute_ptx_pretrain_loss(len(micro_batches))
            stats['ptx_loss'] = ptx_losses

        pi_grad_norm = get_grad_norm_local(self.policy_model)

        if self.policy_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.policy_grad_clip,
                error_if_nonfinite=True,
            )

        self.policy_optimizer.step()
        self.policy_scheduler.step()

        t1 = time.time()
        self.c_policy_update += 1
        # Average over accumulated steps
        step_stats = {k: np.mean(v) for k, v in stats.items() if len(v) > 0}
        step_stats['learning_rate'] = self.policy_optimizer.param_groups[0]['lr']
        step_stats['grad_norm'] = pi_grad_norm.item()
        step_stats['step_time'] = t1 - t0
        return step_stats

    def update_value_model(
        self,
        micro_batches: List[Mapping[Text, torch.Tensor]],
    ) -> Mapping[Text, Any]:
        torch.cuda.empty_cache()
        self.value_optimizer.zero_grad()

        stats = {
            'error': [],
            'loss': [],
            'clipfrac': [],
        }

        t0 = time.time()

        # accumulate gradients over N micro batches
        for batch in micro_batches:
            tokens = batch['tokens'].to(dtype=torch.long, device=self.value_device)
            values = batch['values'].to(dtype=torch.float, device=self.value_device)
            returns = batch['returns'].to(dtype=torch.float, device=self.value_device)
            mask = batch['mask'].to(dtype=torch.bool, device=self.value_device)

            # Given past states, get predicted state value
            vpred = self.value_model(tokens).squeeze(-1)
            assert vpred.shape == returns.shape

            if self.value_clip_eps > 0:
                vpredclipped = torch.clamp(vpred, values - self.value_clip_eps, values + self.value_clip_eps)
                vf_losses1 = torch.square(vpred - returns)
                vf_losses2 = torch.square(vpredclipped - returns)
                vf_losses = 0.5 * torch.max(vf_losses1, vf_losses2)
            else:
                vf_losses = 0.5 * torch.square(vpred - returns)

            # apply loss mask
            assert vf_losses.shape == mask.shape
            vf_loss = masked_mean(vf_losses, mask)

            # scale the loss to account for gradient accumulation
            scaled_loss = vf_loss / len(micro_batches)
            scaled_loss.backward()

            pred_error = masked_mean(torch.square(vpred.detach() - returns.detach()), mask)

            stats['loss'].append(vf_loss.detach().item())
            stats['error'].append(pred_error.detach().item())

            if self.value_clip_eps > 0:
                clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1), mask)
                stats['clipfrac'].append(clipfrac.detach().item())

        value_grad_norm = get_grad_norm_local(self.value_model)

        if self.value_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(),
                max_norm=self.value_grad_clip,
                error_if_nonfinite=True,
            )

        self.value_optimizer.step()
        self.value_scheduler.step()

        t1 = time.time()
        self.c_value_update += 1

        # Average over accumulated steps
        step_stats = {k: np.mean(v) for k, v in stats.items() if len(v) > 0}
        step_stats['learning_rate'] = self.value_optimizer.param_groups[0]['lr']
        step_stats['grad_norm'] = value_grad_norm.item()
        step_stats['step_time'] = t1 - t0
        return step_stats

    def compute_iteration_stats_from_episodes(self, episodes: List[Mapping[Text, torch.Tensor]]) -> Mapping[Text, float]:
        """Aggregate statistics over the current iteration"""

        batch = self.get_batched_transitions(episodes)
        stats = {}
        mask = batch['mask']

        for k in ['values', 'returns']:
            data = batch[k]
            stats[f'objective/{k}_mean'] = masked_mean(data, mask).item()
            stats[f'objective/{k}_var'] = masked_var(data, mask).item()

        for k in ['rewards', 'kl', 'kl_penalties']:
            data = torch.sum(batch[k] * mask.float(), dim=1)
            stats[f'objective/{k}'] = data.mean().item()

        # add more statistics
        stats['objective/scalar_reward'] = torch.mean(batch['scalar_reward']).item()
        stats['objective/entropy'] = masked_mean(batch['entropies'], mask).item()
        stats['elapsed/episode_steps'] = torch.mean(batch['episode_steps']).item()
        stats['elapsed/time'] = torch.sum(batch['episode_time']).item()
        return stats

    def log_selfplay_episode_stats(
        self,
        episodes: List[Mapping[Text, torch.Tensor]],
        log_interval: int,
        sample_interval: int,
        is_training: bool = False,
    ) -> Tuple[List[Mapping[Text, Any]], Mapping[Text, Any]]:
        """Log episode statistics and generated text to tensorboard"""

        tb_tag = 'train_episodes' if is_training else 'val_episodes'
        for episode in episodes:
            mask = episode['mask']
            stats = {
                'episode_steps': episode['episode_steps'].item(),
                'scalar_reward': episode['scalar_reward'].item(),  # normalized reward from reward model
                'episode_time': episode['episode_time'].item(),
            }

            if 'kl' in episode:
                stats['kl'] = masked_sum(episode['kl'], mask).item()

            if is_training:
                self.c_train_episode += 1
            else:
                self.c_val_episode += 1

            episode_count = self.c_train_episode if is_training else self.c_val_episode

            if episode_count % log_interval == 0:
                self.log_stats_to_tensorboard(stats, episode_count, tb_tag)

            # sample generated text
            if episode_count % sample_interval == 0:
                start_t = episode['start_steps']
                prompt_tokens = episode['tokens'][: start_t + 1].tolist()
                response_tokens = episode['actions'][start_t:].tolist()
                prompt_text = self.tokenizer.decode(prompt_tokens)
                response_text = self.tokenizer.decode(response_tokens)

                # Works better when enable 'MarkDown' in tensorboard
                formatted_text = f"**Prompt**: {prompt_text}   <br><br>**Response**: {response_text}   <br><br>**Reward**: {stats['scalar_reward']:.2f}"
                formatted_text = f"{formatted_text}   <br><br>**KL**: {stats['kl']:.2f}" if 'kl' in stats else formatted_text
                self.log_text_to_tensorboard(formatted_text, episode_count, f'{tb_tag}_sample')

    def log_text_to_tensorboard(self, text: str, step: int, tag: str) -> None:
        assert text is not None
        assert tag is not None
        assert step >= 1

        if self.tb_writer is None:
            return

        self.tb_writer.add_text(tag, text, step)

    def log_stats_to_tensorboard(self, stats: Mapping[Text, Any], step: int, tag: str) -> None:
        assert tag is not None
        assert step >= 1

        if self.tb_writer is None:
            return

        for k, v in stats.items():
            if isinstance(v, (int, float)):
                self.tb_writer.add_scalar(f'{tag}/{k}', v, step)

    def state_dict(self) -> Mapping[Text, Any]:
        return {
            "iteration": self.c_iteration,
            "policy_update": self.c_policy_update,
            "value_update": self.c_value_update,
            "train_episode": self.c_train_episode,
            "val_episode": self.c_val_episode,
            "reward_normalizer_state": self.reward_normalizer.state_dict(),
            "kl_coef": self.kl_ctl.value,
        }

    def load_state_dict(self, state: Mapping[Text, Any]) -> None:
        if "iteration" in state:
            self.c_iteration = state["iteration"]
        if "policy_update" in state:
            self.c_policy_update = state["policy_update"]
        if "value_update" in state:
            self.c_value_update = state["value_update"]
        if "train_episode" in state:
            self.c_train_episode = state["train_episode"]
        if "val_episode" in state:
            self.c_val_episode = state["val_episode"]
        if "reward_normalizer_state" in state:
            self.reward_normalizer.load_state_dict(state["reward_normalizer_state"])
        if "kl_coef" in state:
            self.kl_ctl.kl_coef = state["kl_coef"]
