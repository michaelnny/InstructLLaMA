# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Train policy and value models using PPO algorithm (RL) and QLoRA, starting from fine-tuned model and reward model (RM) checkpoints."""

import os
from typing import Tuple, List, Union, Mapping, Text, Any, Callable
from functools import partial
import random
import time
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model_lora import Transformer, LoraModelArgs
from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.models.lora import mark_only_lora_as_trainable
from instruct_llama.configs.rlhf_lora import config as cfg
from instruct_llama.utils.custom_dataset import BlendedDataset, PromptOnlyDataset
from instruct_llama.utils.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.utils.train_helper import (
    create_trace_profiler,
    create_optimizer,
    get_grad_norm_local,
    masked_whiten,
    masked_mean,
    masked_sum,
    split_indices_into_bins,
)
from instruct_llama.utils.logger import create_logger
from instruct_llama.utils.env import PromptEnv
from instruct_llama.utils.normalizer import RunningMeanStd
from instruct_llama.utils.checkpoint import create_lora_checkpoint
from instruct_llama.generation import sample_top_p

logger = create_logger()


def clear_gpu_cache(rank=None):
    torch.cuda.empty_cache()


def convert_model_to_dtype(model: Transformer, compute_dtype):
    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=compute_dtype)


def build_model(
    vocab_size, ckpt_file, device, compute_dtype, not_trainable: bool = True, is_policy: bool = True, strict: bool = True
) -> Transformer:
    assert vocab_size > 0

    model_args = LoraModelArgs.from_model_type(
        model_type=cfg.policy_model_type if is_policy else cfg.reward_model_type,
        # LoRA configurations, note if it's for inference then no need to user lora
        lora_r=0 if not_trainable else cfg.lora_r,
        lora_scaling=0 if not_trainable else cfg.lora_scaling,
        lora_dropout=0 if not_trainable else cfg.lora_dropout,
        # LoRA trainable layers, not need to apply LoRA if not trainable
        lora_attn_query=False if not_trainable else cfg.lora_attn_query,
        lora_attn_key=False if not_trainable else cfg.lora_attn_key,
        lora_attn_value=False if not_trainable else cfg.lora_attn_value,
        lora_attn_proj=False if not_trainable else cfg.lora_attn_proj,
        lora_attn_mlp=False if not_trainable else cfg.lora_attn_mlp,
        # Quantization configurations
        quant_4bit=True if not_trainable else cfg.quant_4bit,  # always quantize frozen model to save GPU RAM
        quant_lora_4bit=False if not_trainable else cfg.quant_lora_4bit,
        quant_4bit_double=True if not_trainable else cfg.quant_4bit_double,
        quant_4bit_type=cfg.quant_4bit_type,
        quant_compute_dtype=compute_dtype,
        # Regular configurations
        head_type='lm_head' if is_policy else 'scalar_head',
        use_cache=False,
        max_seq_len=cfg.max_seq_len,
        max_batch_size=cfg.selfplay_batch_size if is_policy and not not_trainable else 1,
        embed_dropout=0.0 if not_trainable else cfg.embed_dropout,
        attn_dropout=0.0 if not_trainable else cfg.attn_dropout,
        resid_dropout=0.0 if not_trainable else cfg.resid_dropout,
        gradient_checkpointing=False if not_trainable else cfg.gradient_checkpointing,
    )

    model = Transformer(model_args)

    if os.path.exists(ckpt_file):
        logger.info(f'Loading model checkpoint {ckpt_file!r} ...')
        model_state = torch.load(ckpt_file)
        model.load_state_dict(model_state, strict=strict)
        del model_state  # free up CPU RAM

    if device == 'cpu':
        convert_model_to_dtype(model, torch.float32)
    else:
        convert_model_to_dtype(model, compute_dtype)

    if not_trainable:
        for p in model.parameters():
            p.requires_grad = False

    return model.to(device)


def clip_reward(x, max_abs_reward) -> torch.Tensor:
    if max_abs_reward > 0:
        return torch.clamp(x, min=-max_abs_reward, max=max_abs_reward)
    else:
        return x


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

    Args:
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


def get_batched_transitions(micro_batch: List[Mapping[Text, torch.Tensor]], eos_id: int) -> Mapping[Text, torch.Tensor]:
    """Essentially the same as a regular custom collate function for dataloader, except here we don't use dataloader"""
    batch_size = len(micro_batch)
    max_batch_len = max([len(item['tokens']) for item in micro_batch])

    batched_tokens = torch.full((batch_size, max_batch_len), eos_id, dtype=torch.long)
    batched_actions = torch.full((batch_size, max_batch_len), eos_id, dtype=torch.long)
    batched_logprobs = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
    batched_values = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
    batched_returns = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
    batched_advantages = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
    batched_mask = torch.full((batch_size, max_batch_len), 0, dtype=torch.bool)

    for i, item in enumerate(micro_batch):
        seq_len = len(item['tokens'])
        batched_tokens[i, :seq_len] = item['tokens']
        batched_actions[i, :seq_len] = item['actions']
        batched_logprobs[i, :seq_len] = item['logprobs']
        batched_values[i, :seq_len] = item['values']
        batched_returns[i, :seq_len] = item['returns']
        batched_advantages[i, :seq_len] = item['advantages']
        batched_mask[i, :seq_len] = item['mask']

    return {
        'tokens': batched_tokens,
        'actions': batched_actions,
        'logprobs': batched_logprobs,
        'values': batched_values,
        'returns': batched_returns,
        'advantages': batched_advantages,
        'mask': batched_mask,
    }


def find_begin_of_inst_index(input_list, pattern=[518, 25580, 29962]):
    """Find the index of the [INST] token from the given list"""
    pattern_length = len(pattern)
    lst_length = len(input_list)
    for i in range(lst_length - pattern_length + 1):
        if input_list[i : i + pattern_length] == pattern:
            return i

    return -1  # Return -1 if pattern is not found


class PPOAgent:

    """PPO agent for self-play and learning"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        ptx_loader: DataLoader,
        policy_model: Transformer,
        value_model: Transformer,
        policy_optimizer: torch.optim.AdamW,
        policy_scheduler: CosineDecayWithWarmupLRScheduler,
        value_optimizer: torch.optim.AdamW,
        value_scheduler: CosineDecayWithWarmupLRScheduler,
        update_epochs: int,
        train_batch_size: int,
        gradient_accum_steps: int,
        loss_scale: float = 1.0,
        policy_clip_eps: float = 0.2,
        value_clip_eps: float = 0.2,
        clip_kl: float = 0.2,
        kl_coef: float = 0.02,
        entropy_coef: float = 0.0,
        ptx_coef: float = 0.01,
        discount: float = 1.0,
        gae_lambda: float = 0.95,
        policy_grad_clip: float = 2.0,
        value_grad_clip: float = 10.0,
        selfplay_log_interval: int = 10,
        train_log_interval: int = 10,
        policy_device: str = 'cuda',
        value_device: str = 'cuda',
        tb_writer: SummaryWriter = None,
    ):
        self.tokenizer = tokenizer
        self.ptx_loader = ptx_loader
        self.policy_device = policy_device
        self.value_device = value_device

        self.policy_model = policy_model.to(self.policy_device)
        self.policy_optimizer = policy_optimizer
        self.policy_scheduler = policy_scheduler
        self.policy_grad_clip = policy_grad_clip

        self.value_model = value_model.to(self.value_device)
        self.value_optimizer = value_optimizer
        self.value_scheduler = value_scheduler
        self.value_grad_clip = value_grad_clip

        self.policy_model.disable_cache()
        self.value_model.disable_cache()
        self.model_params = self.policy_model.params

        self.update_epochs = update_epochs
        self.train_batch_size = train_batch_size
        self.gradient_accum_steps = gradient_accum_steps
        self.loss_scale = loss_scale
        self.value_clip_eps = value_clip_eps
        self.policy_clip_eps = policy_clip_eps
        self.entropy_coef = entropy_coef
        self.ptx_coef = ptx_coef
        self.clip_kl = clip_kl
        self.kl_coef = kl_coef
        self.discount = discount
        self.gae_lambda = gae_lambda

        self.selfplay_log_interval = selfplay_log_interval
        self.train_log_interval = train_log_interval
        self.tb_writer = tb_writer

        # counters
        self.c_policy_update = 0
        self.c_value_update = 0
        self.c_train_episode = 0
        self.c_val_episode = 0
        self.c_epoch = 0

    @torch.no_grad()
    def run_selfplay(
        self,
        env: PromptEnv,
        num_episodes: int,
        batch_size: int,
        temperature: float,
        top_p: float,
        min_gen_len: int,
        max_gen_len: int,
        normalize_rewards: bool,
        normalize_advantages: bool = False,
        is_training: bool = False,
    ):
        self.policy_model.eval()
        self.value_model.eval()

        batched_episodes = []
        episode_c = 0
        t0 = time.time()

        while episode_c < num_episodes:
            episodes = self.generate_batch_selfplay_episodes(
                env, batch_size, temperature, top_p, max_gen_len, normalize_rewards
            )
            batched_episodes.append(episodes)
            episode_c += len(episodes['terminal_steps'])

        total_act_time = time.time() - t0
        avg_act_time = total_act_time / num_episodes

        logger.info(
            f'Finished generating {episode_c} episodes in {total_act_time:.2f} seconds, average time per episode is {avg_act_time:.2f} second'
        )

        t1 = time.time()
        ppo_transitions = self.get_ppo_transitions_from_batched_episodes(
            batched_episodes, min_gen_len, normalize_advantages, is_training
        )

        total_build_time = time.time() - t1
        avg_build_time = total_build_time / len(ppo_transitions)
        logger.info(
            f'Finished processing {len(ppo_transitions)} episodes in {total_build_time:.2f} seconds, average time per episode is {avg_build_time:.2f} second'
        )

        self.log_selfplay_episode_stats(ppo_transitions, is_training)

        return ppo_transitions

    @torch.no_grad()
    def generate_batch_selfplay_episodes(
        self,
        env: PromptEnv,
        batch_size: int,
        temperature: float,
        top_p: float,
        max_gen_len: int,
        normalize_rewards: bool,
    ) -> Mapping[Text, torch.Tensor]:
        """Run one batch episodes, where the code is adapted from the generation.py module,
        here we also store the intermediate transitions which are required to train the model using the PPO algorithm"""
        assert batch_size >= 1
        assert max_gen_len >= 12

        torch.cuda.empty_cache()
        self.policy_model.enable_cache()

        # randomly sample a batch of prompts
        prompt_tokens = env.reset(batch_size)

        bsz = len(prompt_tokens)
        assert bsz <= self.model_params.max_batch_size, (bsz, self.model_params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        assert min_prompt_len > 2

        total_len = min(self.model_params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device='cuda')
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cuda')

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device='cuda')
        input_text_mask = tokens != pad_id

        # RL agent starts selfplay
        for cur_pos in range(min_prompt_len, total_len):
            output = self.policy_model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits = output[:, -1, :]  # [batch_size, 1, vocab_size]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            next_token = sample_top_p(probs, top_p).reshape(-1)

            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        # start post-selfplay processing
        self.policy_model.disable_cache()
        torch.cuda.empty_cache()

        start_steps = torch.zeros((bsz,), dtype=torch.long, device='cuda')
        terminal_steps = torch.zeros((bsz,), dtype=torch.long, device='cuda')

        # cut tokens to:
        # a. maximum generation length
        # b. eos token
        for i, toks in enumerate(tokens.tolist()):
            start = len(prompt_tokens[i])
            start_steps[i] = start
            _toks = toks[start : start + max_gen_len]

            # cut to max gen len, -1 to avoid out of index
            end_idx = min(start + max_gen_len, total_len) - 1

            # cut to eos token </eos>
            if eos_id in _toks:
                end_idx = start + _toks.index(eos_id)
                assert toks[end_idx] == eos_id
            # else:
            #     # cut to begin of instruction token [INST]
            #     begin_inst_idx = find_begin_of_inst_index(toks)
            #     if begin_inst_idx > 0:
            #         end_idx = start + begin_inst_idx

            assert end_idx >= start and end_idx < total_len and end_idx <= start + max_gen_len
            terminal_steps[i] = end_idx

        # build up loss mask, where we only keep the completions tokens
        # for example, if we have a sequence of:
        # [1, 2, 3, 4, 5, 6, 7, -1, -1]
        # where:
        #   [1, 2, 3, 4] are prompt tokens
        #   [5, 6, 7] are completion tokens
        #   [-1, -1] are padding tokens
        #
        # then the mask will be:
        # [False, False, False, False, True, True, True, False, False]
        mask = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')

        for i, (start_idx, end_idx) in enumerate(zip(start_steps.tolist(), terminal_steps.tolist())):
            mask[i, start_idx : end_idx + 1] = True
            tokens[i, end_idx + 1 :] = eos_id  # replace pad_id

        mask_float = mask.float().cuda()

        # shift one step to left to get the actions taken by the agent, this aligns with the RL transition convention: (s, a, logprob_a)
        actions = torch.full((bsz, total_len), eos_id, dtype=torch.long, device='cuda')
        actions[:, :-1] = tokens[:, 1:]

        # compute log probability for actions
        output = self.policy_model.forward(tokens)
        pi_dist = Categorical(logits=output)
        logprobs = pi_dist.log_prob(actions)
        logprobs *= mask_float

        # Compute environment reward
        env_rewards = env.step(tokens, terminal_steps)
        # Compute pre-token KL penalties for completion tokens
        kl = env.compute_kl_penalties(tokens, actions, logprobs, mask, self.kl_coef)

        if self.clip_kl > 0:
            kl = torch.clamp(kl, min=-self.clip_kl, max=self.clip_kl)

        # Combine environment reward and KL penalties together
        rewards = torch.zeros_like(tokens, dtype=torch.float, device=kl.device)
        for i, idx in enumerate(terminal_steps.tolist()):
            rewards[i, idx] = env_rewards[i]

        rewards -= kl
        rewards *= mask_float.to(kl.device)

        if normalize_rewards:
            rewards = masked_whiten(rewards, mask.to(rewards.device), shift_mean=False)

        episodes = {
            'tokens': tokens.cpu(),
            'actions': actions.cpu(),
            'logprobs': logprobs.cpu(),
            'mask': mask.cpu(),
            'rewards': rewards.cpu(),
            'env_rewards': env_rewards.cpu(),
            'kl': kl.cpu(),
            'start_steps': start_steps.cpu(),
            'terminal_steps': terminal_steps.cpu(),
        }

        return episodes

    @torch.no_grad()
    def get_ppo_transitions_from_batched_episodes(
        self,
        batched_episodes: List[Mapping[Text, torch.Tensor]],
        min_gen_len: int,
        normalize_advantages: bool,
        is_training: bool = False,
    ) -> Tuple[List[Mapping[Text, torch.Tensor]]]:
        if is_training:
            batched_episodes = self.compute_state_values_for_batched_episodes(batched_episodes)

        episodes = self.flatten_batched_episodes(batched_episodes, min_gen_len)

        if is_training:
            episodes = self.compute_masked_returns_and_advantages(episodes, normalize_advantages)

        return episodes

    @torch.no_grad()
    def compute_state_values_for_batched_episodes(
        self, episodes_list: List[Mapping[Text, torch.Tensor]]
    ) -> List[Mapping[Text, torch.Tensor]]:
        processed_episodes = []
        for episodes in episodes_list:
            tokens = episodes['tokens'].to(self.value_device)
            values = self.value_model(tokens).squeeze(-1)  # [batch_size, seq_len]
            mask = episodes['mask'].to(self.value_device)
            values *= mask.float()
            episodes['values'] = values.cpu()
            processed_episodes.append(episodes)

        return processed_episodes

    @torch.no_grad()
    def compute_masked_returns_and_advantages(
        self, episodes: List[Mapping[Text, torch.Tensor]], normalize_advantages: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        results = []

        for episode in episodes:
            values = episode['values']
            rewards = episode['rewards']
            mask = episode['mask']

            r_t = rewards
            v_t = values
            # pad value at t_p1 step to zero
            v_tp1 = torch.zeros_like(v_t, device='cpu')
            v_tp1[:-1] = v_t[1:]

            done_tp1 = torch.zeros_like(v_tp1, dtype=torch.bool, device='cpu')
            done_tp1[-1] = True
            discount_tp1 = (~done_tp1).float() * self.discount

            adv_t = truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, self.gae_lambda)
            return_t = adv_t + v_t

            if normalize_advantages:
                adv_t = masked_whiten(adv_t, mask, dim=0, shift_mean=True)

            episode['returns'] = return_t * mask.float()
            episode['advantages'] = adv_t * mask.float()
            results.append(episode)

        return results

    def flatten_batched_episodes(
        self, episodes_list: List[Mapping[Text, torch.Tensor]], min_len: int
    ) -> List[Mapping[Text, torch.Tensor]]:
        results = []
        skipped = 0
        for episodes in episodes_list:
            # for each episode in current batch
            for i in range(len(episodes['terminal_steps'])):
                start_step = episodes['start_steps'][i]
                terminal_step = episodes['terminal_steps'][i]

                if terminal_step - start_step < min_len:
                    skipped += 1
                    continue

                end = terminal_step + 1  # plus one because high is exclusive

                episode = {
                    'tokens': episodes['tokens'][i, :end],
                    'actions': episodes['actions'][i, :end],
                    'logprobs': episodes['logprobs'][i, :end],
                    'mask': episodes['mask'][i, :end],
                    'rewards': episodes['rewards'][i, :end],
                    'kl': episodes['kl'][i, :end],
                    'env_rewards': episodes['env_rewards'][i],
                    'start_step': start_step,
                    'terminal_steps': terminal_step,
                }

                # only training episodes have values
                if 'values' in episodes:
                    episode['values'] = episodes['values'][i, :end]

                results.append(episode)

        logger.info(f'Skipped {skipped} episodes with completion tokens lesser than {min_len}')

        return results

    def run_ppo_training_steps(self, transitions: List[Mapping[Text, torch.Tensor]]) -> None:
        """
        Uses PPO and the RL agent generated self-play episodes to train the policy and value networks M epochs.
        """
        # Run M epochs to update policy model parameters
        logger.info(
            f'Starting to run {self.update_epochs} epochs over {len(transitions)} episodes to update PPO policy model ...'
        )
        eos_id = self.tokenizer.eos_id

        # make room for policy model
        if self.policy_device == self.value_device:
            self.value_model.to('cpu')

        self.policy_model.disable_cache()
        self.policy_model.to(self.policy_device)
        self.policy_model.train()

        for _ in range(self.update_epochs):
            # Split transitions into micro batches
            batch_indices = split_indices_into_bins(self.train_batch_size, len(transitions), shuffle=True, drop_last=True)
            micro_batches = [get_batched_transitions([transitions[i] for i in indices], eos_id) for indices in batch_indices]

            for i in range(0, len(micro_batches), self.gradient_accum_steps):
                stats = self.update_policy_model(micro_batches[i : i + self.gradient_accum_steps])
                if self.c_policy_update % self.train_log_interval == 0:
                    self.log_train_stats(stats, True)

        # Run M epochs to update state value model parameters, we use two loops
        # because we can't host the two models at the same time with a single GPU
        logger.info(
            f'Starting to run {self.update_epochs} epochs over {len(transitions)} episodes to update PPO value model ...'
        )

        # make room for value model
        if self.policy_device == self.value_device:
            self.policy_model.to('cpu')

        self.value_model.disable_cache()
        self.value_model.to(self.value_device)
        self.value_model.train()

        for _ in range(self.update_epochs):
            # Split transitions into micro batches
            batch_indices = split_indices_into_bins(self.train_batch_size, len(transitions), shuffle=True, drop_last=True)
            micro_batches = [get_batched_transitions([transitions[i] for i in indices], eos_id) for indices in batch_indices]

            for i in range(0, len(micro_batches), self.gradient_accum_steps):
                stats = self.update_value_model(micro_batches[i : i + self.gradient_accum_steps])
                if self.c_value_update % self.train_log_interval == 0:
                    self.log_train_stats(stats, False)

        # restore original device
        self.policy_model.to(self.policy_device)

    def compute_ptx_pretrain_loss(self) -> float:
        total_loss = 0
        for i, (x, y) in enumerate(self.ptx_loader):
            x = x.to(self.policy_device)
            y = y.to(self.policy_device)

            y_pred = self.policy_model(x)
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), reduction='mean')

            loss *= self.ptx_coef
            scaled_loss = loss * self.loss_scale
            scaled_loss.backward()

            total_loss += loss.detach().item()

            if i >= self.gradient_accum_steps:
                break

        return total_loss

    def update_policy_model(
        self,
        micro_batches: List[Mapping[Text, torch.Tensor]],
    ) -> Mapping[Text, Any]:
        clear_gpu_cache()

        self.policy_optimizer.zero_grad()

        stats = {
            'pg_loss': 0,
            'entropy_loss': 0,
            'ptx_loss': 0,
            'total_loss': 0,
            'entropy': 0,
            'approx_kl': 0,
        }

        t0 = time.time()
        # accumulate gradients over N micro batches
        for micro_batch in micro_batches:
            tokens = (micro_batch['tokens']).to(dtype=torch.long, device=self.policy_device)
            actions = (micro_batch['actions']).to(dtype=torch.long, device=self.policy_device)  # Actions are discrete
            behavior_logprobs = (micro_batch['logprobs']).to(dtype=torch.float, device=self.policy_device)
            advantages = (micro_batch['advantages']).to(dtype=torch.float, device=self.policy_device)
            mask = (micro_batch['mask']).to(dtype=torch.float, device=self.policy_device)

            # Given past states, get predicted action probabilities and state value
            pi_logits = self.policy_model(tokens)

            pi_dist = Categorical(logits=pi_logits)
            pi_logprobs = pi_dist.log_prob(actions)
            entropy = pi_dist.entropy()

            assert pi_logprobs.shape == behavior_logprobs.shape

            # Compute PPO clipped surrogate objective
            ratio = torch.exp(pi_logprobs - behavior_logprobs)
            clipped_ratio = torch.clamp(ratio, min=1.0 - self.policy_clip_eps, max=1.0 + self.policy_clip_eps)
            pg_loss = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

            # apply loss mask
            assert entropy.shape == pg_loss.shape == mask.shape
            entropy = masked_mean(entropy, mask)
            pg_loss = masked_mean(pg_loss, mask)

            # Averaging over batch dimension
            pg_loss = torch.mean(pg_loss)
            entropy = torch.mean(entropy)
            entropy_loss = self.entropy_coef * entropy

            # Negative sign to indicate we want to maximize the policy gradient objective function and entropy
            loss = -(pg_loss + entropy_loss)
            scaled_loss = loss * self.loss_scale  # scale the loss to account for gradient accumulation
            scaled_loss.backward()

            # track training statistics
            approx_kl = 0.5 * torch.mean(torch.square(pi_logprobs.detach() - behavior_logprobs.detach()))

            stats['pg_loss'] += pg_loss.detach().item()
            stats['entropy_loss'] += entropy_loss.detach().item()
            stats['total_loss'] += -loss.detach().item()  # use the original ones when logging
            stats['entropy'] += entropy.detach().item()
            stats['approx_kl'] += approx_kl.detach().item()

        # Compute ptx loss, mixing pretraining gradients into PPO,
        # we do it separately because a single backward() call will cause CUDA OOM
        if self.ptx_coef > 0:
            ptx_loss = self.compute_ptx_pretrain_loss()
            stats['ptx_loss'] += ptx_loss
            stats['total_loss'] += ptx_loss

        grad_norm = get_grad_norm_local(self.policy_model)

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
        stats = {k: v / len(micro_batches) for k, v in stats.items()}

        stats['learning_rate'] = self.policy_optimizer.param_groups[0]['lr']
        stats['grad_norm'] = grad_norm.item()
        stats['step_time'] = t1 - t0

        return stats

    def update_value_model(
        self,
        micro_batches: List[Mapping[Text, torch.Tensor]],
    ) -> Mapping[Text, Any]:
        clear_gpu_cache()

        self.value_optimizer.zero_grad()

        stats = {
            'error': 0,
            'loss': 0,
        }

        t0 = time.time()

        # accumulate gradients over N micro batches
        for batch in micro_batches:
            tokens = (batch['tokens']).to(dtype=torch.long, device=self.value_device)
            values = (batch['values']).to(dtype=torch.float, device=self.value_device)
            returns = (batch['returns']).to(dtype=torch.float, device=self.value_device)
            mask = (batch['mask']).to(dtype=torch.float, device=self.value_device)

            # Given past states, get predicted action probabilities and state value
            pred_values = self.value_model(tokens).squeeze(-1)

            assert pred_values.shape == returns.shape

            # Why doing this clipping? Is this really necessary?
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L343C38-L347
            if self.value_clip_eps > 0:
                pred_values_clipped = torch.clamp(pred_values, values - self.value_clip_eps, values + self.value_clip_eps)
                vloss_1 = torch.square(pred_values - returns)
                vloss_2 = torch.square(pred_values_clipped - returns)
                value_loss = 0.5 * torch.max(vloss_1, vloss_2)
            else:
                value_loss = 0.5 * torch.square(pred_values - returns)

            pred_error = torch.square(pred_values - returns)

            # apply loss mask
            assert value_loss.shape == mask.shape

            value_loss = masked_mean(value_loss, mask)
            pred_error = masked_mean(pred_error, mask)

            # Averaging over batch dimension
            loss = torch.mean(value_loss)
            scaled_loss = loss * self.loss_scale  # scale the loss to account for gradient accumulation
            scaled_loss.backward()

            stats['error'] += pred_error.detach().mean().item()
            stats['loss'] += loss.detach().item()

        grad_norm = get_grad_norm_local(self.value_model)

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
        stats = {k: v / len(micro_batches) for k, v in stats.items()}

        stats['learning_rate'] = self.value_optimizer.param_groups[0]['lr']
        stats['grad_norm'] = grad_norm.item()
        stats['step_time'] = t1 - t0

        return stats

    def log_selfplay_episode_stats(
        self,
        episodes: List[Mapping[Text, torch.Tensor]],
        is_training: bool = False,
    ) -> Tuple[List[Mapping[Text, Any]], Mapping[Text, Any]]:
        if is_training:
            self.c_epoch += 1

        episode_prefix = 'selfplay_episodes_train' if is_training else 'selfplay_episodes_val'
        epoch_prefix = 'selfplay_epochs_train' if is_training else 'selfplay_epochs_val'
        episodes_stats = []

        for episode in episodes:
            mask = episode['mask']
            stats = {
                'steps': (episode['terminal_steps'] - episode['start_step']).item(),
                'env_reward': episode['env_rewards'].item(),
                'reward': masked_sum(episode['rewards'], mask, 0).item(),
                'kl': masked_sum(episode['kl'], mask, 0).item(),
            }

            if is_training:
                stats['returns'] = masked_mean(episode['returns'], mask, 0).item()
                stats['advantages'] = masked_mean(episode['advantages'], mask, 0).item()

            episodes_stats.append(stats)

            if is_training:
                self.c_train_episode += 1
                episode_count = self.c_train_episode
            else:
                self.c_val_episode += 1
                episode_count = self.c_val_episode

            if episode_count % self.selfplay_log_interval == 0 and self.tb_writer:
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f'{episode_prefix}/{k}', v, episode_count)

        episodes_reward = [s['reward'] for s in episodes_stats]
        episodes_env_reward = [s['env_reward'] for s in episodes_stats]
        episodes_kl = [s['kl'] for s in episodes_stats]
        episodes_steps = [s['steps'] for s in episodes_stats]

        aggregated_stats = {
            'env_reward_mean': np.mean(episodes_env_reward),
            'env_reward_std': np.std(episodes_env_reward),
            'kl_mean': np.mean(episodes_kl),
            'kl_std': np.std(episodes_kl),
            'reward_mean': np.mean(episodes_reward),
            'reward_std': np.std(episodes_reward),
            'episode_steps': np.mean(episodes_steps),
        }

        if is_training:
            aggregated_stats['returns_mean'] = np.mean([s['returns'] for s in episodes_stats])
            aggregated_stats['advantages_mean'] = np.mean([s['advantages'] for s in episodes_stats])

        prefix = 'Train epoch self-play statistics' if is_training else 'Validation epoch self-play statistics'
        logger.info(f'{prefix}: {aggregated_stats}')

        if self.tb_writer:
            for k, v in aggregated_stats.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f'{epoch_prefix}/{k}', v, self.c_epoch)

    def log_train_stats(self, stats, is_policy: bool = False) -> None:
        tb_prefix = 'ppo_policy' if is_policy else 'ppo_value'
        step_count = self.c_policy_update if is_policy else self.c_value_update
        if self.tb_writer is not None:
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f'{tb_prefix}/{k}', v, step_count)

        prefix = 'Train policy statistics' if is_policy else 'Train value statistics'
        logger.info(f'{prefix} steps {step_count}\n{stats}')


def main():
    assert cfg.train_log_interval >= 1
    assert cfg.selfplay_log_interval >= 1
    assert cfg.gradient_accum_steps >= 1
    assert 0 < cfg.loss_scale <= 1
    assert cfg.train_batch_size >= 1
    assert cfg.ckpt_interval >= 1
    assert cfg.train_episodes_per_epoch >= (cfg.gradient_accum_steps * cfg.train_batch_size)

    batch_size = int(cfg.train_batch_size * cfg.gradient_accum_steps)

    if not os.path.exists(cfg.sft_ckpt_file):
        raise ValueError(f'Invalid SFT model checkpoint {cfg.sft_ckpt_file!r}, aborting ...')
    if not os.path.exists(cfg.rm_ckpt_file):
        raise ValueError(f'Invalid RM model checkpoint {cfg.rm_ckpt_file!r}, aborting ...')
    if not os.path.exists(cfg.policy_ckpt_file):
        raise ValueError(f'Invalid policy model checkpoint {cfg.policy_ckpt_file!r}, aborting ...')
    if not os.path.exists(cfg.value_ckpt_file):
        raise ValueError(f'Invalid value model checkpoint {cfg.value_ckpt_file!r}, aborting ...')

    if not (torch.version.cuda and torch.cuda.is_bf16_supported()):
        raise RuntimeError('The script only supports training using CUDA and torch.bfloat16, but GPU does not support it.')

    if not any(['cuda' in k for k in (cfg.env_device, cfg.policy_device, cfg.value_device)]):
        raise ValueError('Bitsandbytes 4bit quantization only works with CUDA, aborting ...')

    # --------------- Load datasets ---------------

    logger.info('Loading datasets ...')

    tokenizer = Tokenizer(cfg.tokenizer_file)
    vocab_size = tokenizer.vocab_size

    # Our custom IterableDatasets already have sharding and shuffle mechanism implemented
    cuda_kwargs = {
        'num_workers': cfg.dataloader_workers,
        'batch_size': cfg.train_batch_size,
        'pin_memory': False,
        'shuffle': False,
        'sampler': None,
    }

    train_ptx_dataset = BlendedDataset(
        data_sources=cfg.train_ptx_datasources,
        max_seq_len=cfg.max_seq_len,
        seed=cfg.seed,
    )

    train_ptx_loader = DataLoader(train_ptx_dataset, **cuda_kwargs)

    logger.info(f'PTX pretrain dataset metadata:\n{train_ptx_dataset.get_metadata()}')

    train_prompt_dataset = PromptOnlyDataset(
        data_sources=cfg.train_prompt_datasources,
        min_seq_len=cfg.min_prompt_len,
        max_seq_len=cfg.max_prompt_len,
        max_samples=cfg.max_train_samples,
        seed=cfg.seed,
    )

    logger.info(f'Train prompt dataset metadata:\n{train_prompt_dataset.get_metadata()}')

    val_prompt_dataset = PromptOnlyDataset(
        data_sources=cfg.val_prompt_datasources,
        min_seq_len=cfg.min_prompt_len,
        max_seq_len=cfg.max_prompt_len,
        max_samples=cfg.max_val_samples,
        seed=cfg.seed,
    )

    logger.info(f'Validation prompt dataset metadata:\n{val_prompt_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    compute_dtype = torch.bfloat16
    torch.set_default_dtype(compute_dtype)

    logger.info('Initializing SFT and RM models ...')
    sft_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.sft_ckpt_file,
        device=cfg.env_device,
        compute_dtype=compute_dtype,
        not_trainable=True,
        is_policy=True,
    )
    reward_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.rm_ckpt_file,
        device=cfg.env_device,
        compute_dtype=compute_dtype,
        not_trainable=True,
        is_policy=False,
    )

    clip_reward_fn = partial(clip_reward, max_abs_reward=cfg.clip_env_reward)
    train_env = PromptEnv(
        prompt_dataset=train_prompt_dataset,
        reward_model=reward_model,
        sft_model=sft_model,
        normalize_reward=cfg.normalize_env_rewards,
        normalizer_ckpt=cfg.rm_norm_ckpt_file,
        clip_reward_fn=clip_reward_fn,
        device=cfg.env_device,
    )
    eval_env = PromptEnv(
        prompt_dataset=val_prompt_dataset,
        reward_model=reward_model,
        sft_model=sft_model,
        normalize_reward=cfg.normalize_env_rewards,
        normalizer_ckpt=cfg.rm_norm_ckpt_file,
        clip_reward_fn=clip_reward_fn,
        device=cfg.env_device,
    )

    logger.info('Initializing PPO policy and value models ...')

    # Load model checkpoint using strict=False,
    # because there are missing keys due to LoRA weights not contained in checkpoint state
    policy_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.policy_ckpt_file,
        device=cfg.policy_device,
        compute_dtype=compute_dtype,
        not_trainable=False,
        is_policy=True,
        strict=False,
    )

    value_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.value_ckpt_file,
        device=cfg.value_device,
        compute_dtype=compute_dtype,
        not_trainable=False,
        is_policy=False,
        strict=False,
    )

    mark_only_lora_as_trainable(policy_model, train_bias=cfg.train_bias, train_head=cfg.train_head)
    mark_only_lora_as_trainable(value_model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    max_train_steps = int(cfg.update_epochs * (cfg.max_episodes / batch_size))
    num_epochs = cfg.max_episodes // cfg.train_episodes_per_epoch

    policy_optimizer = create_optimizer(
        model=policy_model,
        lr=cfg.policy_init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        paged_adamw=cfg.use_paged_adamw,
    )

    policy_scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=policy_optimizer,
        init_lr=cfg.policy_init_lr,
        max_lr=cfg.policy_max_lr,
        min_lr=cfg.policy_min_lr,
        warmup_steps=cfg.policy_warmup_steps,
        max_decay_steps=max_train_steps,
    )
    value_optimizer = create_optimizer(
        model=value_model,
        lr=cfg.value_init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        paged_adamw=cfg.use_paged_adamw,
    )
    value_scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=value_optimizer,
        init_lr=cfg.value_init_lr,
        max_lr=cfg.value_max_lr,
        min_lr=cfg.value_min_lr,
        warmup_steps=cfg.value_warmup_steps,
        max_decay_steps=max_train_steps,
    )

    ppo_agent = PPOAgent(
        tokenizer=tokenizer,
        ptx_loader=train_ptx_loader,
        policy_model=policy_model,
        policy_optimizer=policy_optimizer,
        policy_scheduler=policy_scheduler,
        value_model=value_model,
        value_optimizer=value_optimizer,
        value_scheduler=value_scheduler,
        update_epochs=cfg.update_epochs,
        train_batch_size=cfg.train_batch_size,
        gradient_accum_steps=cfg.gradient_accum_steps,
        loss_scale=cfg.loss_scale,
        policy_clip_eps=cfg.policy_clip_eps,
        value_clip_eps=cfg.value_clip_eps,
        clip_kl=cfg.clip_kl,
        kl_coef=cfg.kl_coef,
        entropy_coef=cfg.entropy_coef,
        ptx_coef=cfg.ptx_coef,
        discount=cfg.discount,
        gae_lambda=cfg.gae_lambda,
        policy_grad_clip=cfg.policy_grad_clip,
        value_grad_clip=cfg.value_grad_clip,
        selfplay_log_interval=cfg.selfplay_log_interval,
        train_log_interval=cfg.train_log_interval,
        policy_device=cfg.policy_device,
        value_device=cfg.value_device,
        tb_writer=SummaryWriter(os.path.join(cfg.log_dir, cfg.policy_model_type)) if cfg.use_tensorboard else None,
    )

    # --------------- Start Training ---------------

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.ckpt_dir, 'policy'), exist_ok=True)
    os.makedirs(os.path.join(cfg.ckpt_dir, 'value'), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        logger.info(f'Epoch {epoch}')
        logger.info(f'Starting to generate {cfg.train_episodes_per_epoch} train episodes ...')

        transitions = ppo_agent.run_selfplay(
            env=train_env,
            num_episodes=cfg.train_episodes_per_epoch,
            batch_size=cfg.selfplay_batch_size,
            temperature=cfg.train_temperature,
            top_p=cfg.train_top_p,
            min_gen_len=cfg.min_gen_len,
            max_gen_len=cfg.max_seq_len - cfg.max_prompt_len,
            normalize_rewards=cfg.normalize_rewards,
            normalize_advantages=cfg.normalize_advantages,
            is_training=True,
        )

        # poor solution to swap model between devices, so the training can run without CUDA OOM
        if cfg.env_device != 'cpu' and cfg.env_device == cfg.policy_device or cfg.env_device == cfg.value_device:
            reward_model.to('cpu')
            sft_model.to('cpu')

        logger.info(f'Starting to train the agent using {len(transitions)} selfplay episodes ...')

        torch.cuda.empty_cache()
        ppo_agent.run_ppo_training_steps(transitions=transitions)

        if cfg.env_device != 'cpu' and cfg.env_device == cfg.policy_device or cfg.env_device == cfg.value_device:
            reward_model.to(cfg.env_device)
            sft_model.to(cfg.env_device)

        # regular checkpointing
        if epoch % cfg.ckpt_interval == 0:
            create_lora_checkpoint(
                policy_model,
                os.path.join(cfg.ckpt_dir, f'policy/lora_{cfg.policy_model_type}-epoch-{epoch}.pth'),
                cfg.train_bias,
                cfg.train_head,
            )
            create_lora_checkpoint(
                value_model,
                os.path.join(cfg.ckpt_dir, f'value/lora_{cfg.reward_model_type}-epoch-{epoch}.pth'),
                cfg.train_bias,
                cfg.train_head,
            )

        # validation episodes
        if cfg.var_interval > 0 and epoch % cfg.var_interval == 0:
            logger.info(f'Starting to generate {cfg.val_episodes_per_epoch} validation episodes ...')
            _ = ppo_agent.run_selfplay(
                env=eval_env,
                num_episodes=cfg.val_episodes_per_epoch,
                batch_size=cfg.selfplay_batch_size,
                temperature=cfg.val_temperature,
                top_p=cfg.val_top_p,
                min_gen_len=cfg.min_gen_len,
                max_gen_len=cfg.max_seq_len - cfg.max_prompt_len,
                normalize_rewards=cfg.normalize_rewards,
                normalize_advantages=False,  # don't compute advantages during inference
                is_training=False,
            )

    # training is done ...show some training stats.
    logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
