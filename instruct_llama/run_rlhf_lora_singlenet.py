# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Train policy model with value head using PPO algorithm (RL) and QLoRA, starting from fine-tuned model and reward model (RM) checkpoints."""

import os
from typing import Tuple, Optional, List, Union, Mapping, Dict, Text, Any, Callable
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
from instruct_llama.configs.rlhf_lora_singlenet import config as cfg
from instruct_llama.utils.custom_dataset import BlendedDataset, PromptOnlyDataset
from instruct_llama.utils.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.utils.train_helper import (
    create_optimizer,
    get_grad_norm_local,
    masked_whiten,
    masked_mean,
    masked_sum,
    split_indices_into_bins,
)
from instruct_llama.utils.logger import create_logger
from instruct_llama.utils.env import PromptEnv
from instruct_llama.utils.checkpoint import create_lora_checkpoint
from instruct_llama.generation import sample_top_p
from instruct_llama.utils.rl_ppo import (
    build_model,
    clip_reward,
    truncated_generalized_advantage_estimation,
    find_begin_of_pattern,
    AdaptiveKLController,
)


logger = create_logger()


def clear_gpu_cache():
    torch.cuda.empty_cache()


class PPOAgent:
    """PPO agent for self-play and learning, using a single policy network with value head"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        ptx_loader: DataLoader,
        policy_model: Transformer,
        policy_optimizer: torch.optim.AdamW,
        policy_scheduler: CosineDecayWithWarmupLRScheduler,
        ppo_update_epochs: int,
        train_batch_size: int,
        gradient_accum_steps: int,
        loss_scale: float,
        policy_clip_eps: float,
        value_clip_eps: float,
        entropy_coef: float,
        value_coef: float,
        ptx_coef: float,
        discount: float,
        gae_lambda: float,
        grad_clip: float,
        truncate_token: Optional[Tuple[str]],
        truncate_penalty_value: float,
        selfplay_log_interval: int = 10,
        train_log_interval: int = 5,
        policy_device: str = 'cuda',
        kl_ctl: AdaptiveKLController = None,
        tb_writer: SummaryWriter = None,
    ):
        assert 0 < loss_scale <= 1, loss_scale
        assert ppo_update_epochs >= 1, ppo_update_epochs
        assert train_batch_size >= 1, train_batch_size
        assert gradient_accum_steps >= 1, gradient_accum_steps
        assert 0 < policy_clip_eps < 0.5, policy_clip_eps
        assert 0 <= value_clip_eps < 0.5, value_clip_eps
        assert 0 <= entropy_coef < 0.1, entropy_coef
        assert 0 <= ptx_coef, ptx_coef
        assert 0.9 < discount <= 1, discount
        assert 0.9 < gae_lambda <= 1, gae_lambda
        assert grad_clip >= 0, grad_clip
        if truncate_token is not None:
            assert isinstance(truncate_token, Tuple), truncate_token
        assert truncate_penalty_value <= 0, truncate_penalty_value

        self.tokenizer = tokenizer
        self.ptx_loader = ptx_loader
        self.policy_device = policy_device

        self.policy_model = policy_model.to(self.policy_device)
        self.policy_optimizer = policy_optimizer
        self.policy_scheduler = policy_scheduler
        self.grad_clip = grad_clip

        self.policy_model.disable_cache()
        self.model_params = self.policy_model.params

        self.ppo_update_epochs = ppo_update_epochs
        self.train_batch_size = train_batch_size
        self.gradient_accum_steps = gradient_accum_steps
        self.loss_scale = loss_scale
        self.value_clip_eps = value_clip_eps
        self.policy_clip_eps = policy_clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ptx_coef = ptx_coef

        self.discount = discount
        self.gae_lambda = gae_lambda

        self.kl_ctl: AdaptiveKLController = kl_ctl
        self.truncate_token = truncate_token
        self.truncate_penalty_value = truncate_penalty_value
        self.apply_truncate_penalty = True if self.truncate_token is not None and self.truncate_penalty_value < 0 else False

        self.selfplay_log_interval = selfplay_log_interval
        self.train_log_interval = train_log_interval
        self.tb_writer = tb_writer

        # counters
        self.c_policy_update = 0
        self.c_train_episode = 0
        self.c_val_episode = 0

    @torch.no_grad()
    def run_selfplay(
        self,
        epoch: int,
        env: PromptEnv,
        num_episodes: int,
        batch_size: int,
        temperature: float,
        top_p: float,
        min_gen_len: int,
        max_gen_len: int,
        selfplay_sample_interval: int,
        is_training: bool = False,
    ) -> List[Dict[Mapping, torch.Tensor]]:
        self.policy_model.eval()
        batched_episodes = []

        # guarantee that we'll have at least M episodes
        episode_c = 0
        discard_c = 0
        while episode_c < num_episodes:
            episodes = self.generate_batch_selfplay_episodes(
                env, batch_size, temperature, top_p, min_gen_len, max_gen_len, is_training
            )
            batched_episodes.append(episodes)
            current_bs = len(episodes['terminal_steps'])
            episode_c += current_bs
            discard_c += batch_size - current_bs

        if discard_c > 0:
            logger.info(f'Discarded {discard_c} episodes with response tokens lesser than {min_gen_len}')

        episodes = self.flatten_batched_episodes(batched_episodes)
        self.log_selfplay_episode_stats(epoch, episodes, selfplay_sample_interval, is_training)
        return episodes

    @torch.no_grad()
    def generate_batch_selfplay_episodes(
        self,
        env: PromptEnv,
        batch_size: int,
        temperature: float,
        top_p: float,
        min_gen_len: int,
        max_gen_len: int,
        is_training: bool = False,
    ) -> Mapping[Text, torch.Tensor]:
        """Run one batch episodes, where the code is adapted from the generation.py module,
        here we also store the intermediate transitions which are required to train the model using the PPO algorithm"""
        assert min_gen_len >= 1
        assert max_gen_len <= self.model_params.max_seq_len
        assert batch_size >= 8

        self.policy_model.enable_cache()
        torch.cuda.empty_cache()

        # sample a batch of prompts randomly
        prompt_tokens = env.reset(batch_size)

        bsz = len(prompt_tokens)
        assert bsz <= self.model_params.max_batch_size, (bsz, self.model_params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        assert min_prompt_len > 6
        total_len = min(self.model_params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.policy_device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.policy_device)

        # store state values from the policy value-head, this is referred as the 'state value' in RL terminology
        token_values = torch.full((bsz, total_len), 0.0, dtype=torch.float, device=self.policy_device)
        # store log probability for actions, this is referred as the 'behavior policy' in RL terminology
        token_logprobs = torch.full((bsz, total_len), 0.0, dtype=torch.float, device=self.policy_device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.policy_device)
        input_text_mask = tokens != pad_id

        t0 = time.time()

        # RL agent starts selfplay
        for cur_pos in range(min_prompt_len, total_len):
            output = self.policy_model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits = output['policy_head'][:, -1, :]  # [batch_size, vocab_size]
            state_value = output['value_head'][:, -1, :].squeeze()  # [batch_size]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            next_token = sample_top_p(probs, top_p).reshape(-1)  # [batch_size]

            token_logprob = torch.gather(torch.log_softmax(logits, dim=-1), dim=1, index=next_token.unsqueeze(1)).squeeze(
                1
            )  # [batch_size]

            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            state_value = torch.where(input_text_mask[:, cur_pos], token_values[:, cur_pos], state_value)
            token_values[:, cur_pos] = state_value

            token_logprob = torch.where(input_text_mask[:, cur_pos], token_logprobs[:, cur_pos], token_logprob)
            token_logprobs[:, cur_pos] = token_logprob

            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        # start post-selfplay processing
        self.policy_model.disable_cache()
        torch.cuda.empty_cache()

        start_steps = torch.zeros((bsz,), dtype=torch.long, device=self.policy_device)
        terminal_steps = torch.zeros((bsz,), dtype=torch.long, device=self.policy_device)
        truncate_steps = torch.zeros((bsz,), dtype=torch.long, device=self.policy_device)

        # cut tokens
        for i, toks in enumerate(tokens.tolist()):
            start_idx = len(prompt_tokens[i])
            start_steps[i] = start_idx
            res_toks = toks[start_idx:]  # response tokens

            # cut to max gen len
            end_idx = len(toks) - 1

            # cut to eos token '</s>'
            if eos_id in res_toks:
                end_idx = start_idx + res_toks.index(eos_id)
                assert toks[end_idx] == eos_id

            if end_idx > start_idx and end_idx < total_len:
                terminal_steps[i] = end_idx

            # looking for special tokens like '[INST]' and something alike, so we can later add a penalty score
            truncate_idx = 0
            if self.apply_truncate_penalty:
                for truncate_token in self.truncate_token:
                    truncate_token_ids = self.tokenizer.encode(truncate_token)
                    pen_start_idx = find_begin_of_pattern(res_toks, truncate_token_ids)
                    if pen_start_idx > 0:
                        truncate_idx = start_idx + pen_start_idx
                        if toks[truncate_idx : truncate_idx + len(truncate_token_ids)] != truncate_token_ids:
                            truncate_idx = 0
                        break

            if truncate_idx > 0 and truncate_idx > start_idx:
                truncate_steps[i] = truncate_idx

        # filter episodes where length of response tokens are too short
        valid_episodes = ((terminal_steps - start_steps) >= min_gen_len).to(self.policy_device)
        num_discard = (~valid_episodes).sum().item()
        if num_discard > 0:
            tokens = tokens[valid_episodes, ...]
            token_values = token_values[valid_episodes, ...]
            token_logprobs = token_logprobs[valid_episodes, ...]
            start_steps = start_steps[valid_episodes]
            terminal_steps = terminal_steps[valid_episodes]
            truncate_steps = truncate_steps[valid_episodes]

        # build up loss mask
        # for example, if we have a sequence of:
        # [1, 2, 3, 4, 5, 6, 7, -1, -1]
        # where:
        #   [1, 2, 3, 4] are prompt tokens
        #   [5, 6, 7] are response tokens
        #   [-1, -1] are padding tokens
        #
        # then the mask will be:
        # [False, False, False, True, True, True, True, False, False]
        mask = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')
        for i, (start_idx, end_idx) in enumerate(zip(start_steps.tolist(), terminal_steps.tolist())):
            mask[i, start_idx : end_idx + 1] = True  # +1 because high is exclusive

        # replace pad ids
        tokens = torch.where(tokens == pad_id, eos_id, tokens)
        mask_float = mask.float().cpu()

        # shift one step to left to get the actions taken by the agent, this aligns with the RL transition convention: (s_t, a_t, logprob_a_t, v_t)
        actions = torch.full((len(tokens), total_len), eos_id, dtype=torch.long, device='cpu')
        actions[:, :-1] = tokens[:, 1:].clone().cpu()

        logprobs = torch.zeros_like(token_logprobs, device='cpu')
        logprobs[:, :-1] = token_logprobs[:, 1:].clone().cpu()
        logprobs *= mask_float

        # similar for the state values, as we store it for next 'state' during self-play
        values = torch.zeros_like(token_values, dtype=torch.float, device='cpu')
        values[:, :-1] = token_values[:, 1:].clone().cpu()
        values *= mask_float

        # Compute environment reward using RM Model
        scalar_rewards = env.step(tokens, terminal_steps).to('cpu')

        # Compute pre-token KL penalties for response tokens
        kl = env.compute_kl_penalties(tokens, actions, logprobs)
        kl = kl.cpu() * mask_float

        # Update adaptive KL controller
        if is_training:
            self.kl_ctl.update(kl.sum(dim=1).mean().item(), len(terminal_steps))

        # RM model reward are zero except for terminal step
        rewards = torch.zeros_like(tokens, dtype=torch.float, device='cpu')
        rewards[torch.arange(len(terminal_steps)), terminal_steps] = scalar_rewards

        # add pre-token KL penalties for the response tokens
        kl_penalties = -self.kl_ctl.value * kl
        rewards += kl_penalties

        # add penalties to truncated tokens in the response
        if self.apply_truncate_penalty:
            truncate_masks = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')
            for i, (t, end_t) in enumerate(zip(truncate_steps.tolist(), terminal_steps.tolist())):
                if t > 0:
                    truncate_masks[i, t : end_t + 1] = True
            truncate_penalties = truncate_masks.float() * self.truncate_penalty_value
            rewards = torch.where(truncate_masks, self.truncate_penalty_value, rewards)
        else:
            truncate_penalties = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')

        t1 = time.time()

        episodes = {
            'tokens': tokens.cpu(),
            'actions': actions.cpu(),
            'logprobs': logprobs.cpu(),
            'values': values.cpu(),
            'mask': mask.cpu(),
            'rewards': rewards.cpu(),
            'scalar_rewards': scalar_rewards.cpu(),  # the remaining are for logging only
            'kl': kl.cpu(),
            'kl_coef': self.kl_ctl.value,
            'kl_penalties': kl_penalties.cpu(),
            'truncate_penalties': truncate_penalties.cpu(),
            'episode_time': (t1 - t0) / len(terminal_steps),
            'start_steps': start_steps.cpu(),
            'terminal_steps': terminal_steps.cpu(),
        }

        return episodes

    def flatten_batched_episodes(self, episodes_list: List[Mapping[Text, torch.Tensor]]) -> List[Mapping[Text, torch.Tensor]]:
        """Turn a list of batched episodes into a flat list of episodes"""
        results = []
        for episodes in episodes_list:  # for each batch
            for i in range(len(episodes['terminal_steps'])):  # for each episode in current batch
                start_step = episodes['start_steps'][i]
                terminal_step = episodes['terminal_steps'][i]
                end = terminal_step + 1  # +1 because high is exclusive
                episode = {
                    'tokens': episodes['tokens'][i, :end],
                    'actions': episodes['actions'][i, :end],
                    'logprobs': episodes['logprobs'][i, :end],
                    'values': episodes['values'][i, :end],
                    'mask': episodes['mask'][i, :end],
                    'scalar_reward': episodes['scalar_rewards'][i],
                    'rewards': episodes['rewards'][i, :end],
                    'kl': episodes['kl'][i, :end],
                    'kl_coef': episodes['kl_coef'],
                    'kl_penalties': episodes['kl_penalties'][i, :end],
                    'truncate_penalties': episodes['truncate_penalties'][i, :end],
                    'episode_time': episodes['episode_time'],
                    'start_step': start_step,
                    'terminal_step': terminal_step,
                }

                results.append(episode)

        return results

    @torch.no_grad()
    def prepare_ppo_transitions(
        self,
        batch: List[Mapping[Text, torch.Tensor]],
        whiten_rewards: bool,
        whiten_advantages: bool,
    ) -> None:
        """Compute GAE advantages for the batch, and optionally normalize reward and advantages wile doing so"""
        batch_size = len(batch)
        max_batch_len = max([len(item['tokens']) for item in batch])

        batched_rewards = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_values = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_returns = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_advantages = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_mask = torch.full((batch_size, max_batch_len), 0, dtype=torch.bool)

        # batch the rewards so we can later normalize it if needed
        for i, item in enumerate(batch):
            seq_len = len(item['tokens'])
            batched_mask[i, :seq_len] = item['mask']
            batched_rewards[i, :seq_len] = item['rewards']
            batched_values[i, :seq_len] = item['values']

        if whiten_rewards:
            batched_rewards = masked_whiten(batched_rewards, batched_mask, shift_mean=False)

        # compute GAE advantages one episode at a time
        for i, item in enumerate(batch):
            seq_len = len(item['tokens'])
            returns, advantages = self.compute_masked_returns_and_advantages(
                batched_values[i, :seq_len], batched_rewards[i, :seq_len], batched_mask[i, :seq_len]
            )
            batched_returns[i, :seq_len] = returns
            batched_advantages[i, :seq_len] = advantages

        if whiten_advantages:
            batched_advantages = masked_whiten(batched_advantages, batched_mask, shift_mean=True)

        # add the computed GAE advantages and returns back to the episode samples
        for i, item in enumerate(batch):
            seq_len = len(item['tokens'])
            item['returns'] = batched_returns[i, :seq_len]
            item['advantages'] = batched_advantages[i, :seq_len]

    @torch.no_grad()
    def compute_masked_returns_and_advantages(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages for a single episode"""

        r_t = rewards * mask.float()
        v_t = values * mask.float()

        # make sure state value for the terminal step is zero
        v_t[-1] = 0
        done_tp1 = torch.zeros_like(v_t, dtype=torch.bool, device='cpu')
        # mark terminal step, note we mark last two steps since it's for 't plus 1'
        done_tp1[-2:] = True

        # pad value at t_p1 step to zero
        v_tp1 = torch.zeros_like(v_t, device='cpu')
        v_tp1[:-1] = v_t[1:]

        discount_tp1 = (~done_tp1).float() * self.discount

        adv_t = truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, self.gae_lambda)
        return_t = adv_t + v_t

        return_t *= mask.float()
        adv_t *= mask.float()
        return return_t, adv_t

    @torch.no_grad()
    def get_batched_transitions(
        self,
        batch: List[Mapping[Text, torch.Tensor]],
    ) -> Mapping[Text, torch.Tensor]:
        """Essentially the same as a regular custom collate function for dataloader, except here we also compute GAE advantages for the batch"""
        batch_size = len(batch)
        max_batch_len = max([len(item['tokens']) for item in batch])
        eos_id = self.tokenizer.eos_id

        batched_tokens = torch.full((batch_size, max_batch_len), eos_id, dtype=torch.long)
        batched_actions = torch.full((batch_size, max_batch_len), eos_id, dtype=torch.long)
        batched_logprobs = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_values = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_returns = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_advantages = torch.full((batch_size, max_batch_len), 0.0, dtype=torch.float)
        batched_mask = torch.full((batch_size, max_batch_len), 0, dtype=torch.bool)

        for i, item in enumerate(batch):
            seq_len = len(item['tokens'])
            batched_tokens[i, :seq_len] = item['tokens']
            batched_actions[i, :seq_len] = item['actions']
            batched_logprobs[i, :seq_len] = item['logprobs']
            batched_values[i, :seq_len] = item['values']
            batched_mask[i, :seq_len] = item['mask']
            batched_returns[i, :seq_len] = item['returns']
            batched_advantages[i, :seq_len] = item['advantages']

        return {
            'tokens': batched_tokens,
            'actions': batched_actions,
            'logprobs': batched_logprobs,
            'values': batched_values,
            'returns': batched_returns,
            'advantages': batched_advantages,
            'mask': batched_mask,
        }

    def run_ppo_training_steps(
        self,
        episodes: List[Mapping[Text, torch.Tensor]],
        whiten_rewards: bool,
        whiten_advantages: bool,
    ) -> None:
        """
        Uses PPO and the RL agent generated selfplay episodes to train the policy network.
        """

        # Compute GAE advantages once over all episodes, since the same transitions will be used multiple times
        self.prepare_ppo_transitions(episodes, whiten_rewards, whiten_advantages)

        # Run M epochs to update policy model
        self.policy_model.disable_cache()
        self.policy_model.train()

        for _ in range(self.ppo_update_epochs):
            # Split episodes into micro batches
            batch_indices = split_indices_into_bins(self.train_batch_size, len(episodes), shuffle=True, drop_last=True)
            micro_batches = [self.get_batched_transitions([episodes[i] for i in indices]) for indices in batch_indices]

            for i in range(0, len(micro_batches), self.gradient_accum_steps):
                stats = self.update_policy_model(micro_batches[i : i + self.gradient_accum_steps])
                if self.c_policy_update % self.train_log_interval == 0:
                    self.log_train_stats(stats)

    def compute_ptx_pretrain_loss(self) -> List[float]:
        losses = []
        if self.ptx_loader is None:
            return losses

        for i, (x, y) in enumerate(self.ptx_loader):
            x = x.to(self.policy_device)
            y = y.to(self.policy_device)

            output = self.policy_model(x)
            y_pred = output['policy_head']
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), reduction='mean')

            loss *= self.ptx_coef
            scaled_loss = loss * self.loss_scale
            scaled_loss.backward()

            losses.append(loss.detach().item())
            if i >= self.gradient_accum_steps:
                break

        return losses

    def update_policy_model(
        self,
        micro_batches: List[Mapping[Text, torch.Tensor]],
    ) -> Mapping[Text, Any]:
        clear_gpu_cache()

        self.policy_optimizer.zero_grad()

        stats = {
            'pg_loss': [],
            'entropy': [],
            'entropy_loss': [],
            'ptx_loss': [],
            'value_error': [],
            'value_loss': [],
        }

        t0 = time.time()
        # accumulate gradients over N micro batches
        for batch in micro_batches:
            tokens = (batch['tokens']).to(dtype=torch.long, device=self.policy_device)
            actions = (batch['actions']).to(dtype=torch.long, device=self.policy_device)
            behavior_logprobs = (batch['logprobs']).to(dtype=torch.float, device=self.policy_device)
            advantages = (batch['advantages']).to(dtype=torch.float, device=self.policy_device)
            values = (batch['values']).to(dtype=torch.float, device=self.policy_device)
            returns = (batch['returns']).to(dtype=torch.float, device=self.policy_device)
            mask = (batch['mask']).to(dtype=torch.float, device=self.policy_device)

            # Given past states, get predicted action probabilities and state value
            output = self.policy_model(tokens)

            pi_logits = output['policy_head']  # [batch_size, seqlen, vocab_size]
            pred_values = output['value_head'].squeeze(-1)  # [batch_size, seqlen]

            pi_logprobs = torch.gather(torch.log_softmax(pi_logits, dim=-1), dim=2, index=actions.unsqueeze(2)).squeeze(2)
            pd = torch.softmax(pi_logits, dim=-1)
            entropy = torch.logsumexp(pi_logits, dim=-1) - torch.sum(pd * pi_logits, dim=-1)

            assert pi_logprobs.shape == behavior_logprobs.shape  # [batch_size, seqlen]

            # Compute PPO clipped surrogate objective
            ratio = torch.exp(pi_logprobs - behavior_logprobs)
            clipped_ratio = torch.clamp(ratio, min=1.0 - self.policy_clip_eps, max=1.0 + self.policy_clip_eps)
            pg_loss = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

            # PPO state value loss
            # Is this value clipping really necessary?
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L343C38-L347
            if self.value_clip_eps > 0:
                pred_values_clipped = torch.clamp(
                    pred_values, min=values - self.value_clip_eps, max=values + self.value_clip_eps
                )
                vloss_1 = torch.square(pred_values - returns)
                vloss_2 = torch.square(pred_values_clipped - returns)
                value_loss = 0.5 * torch.max(vloss_1, vloss_2)
            else:
                value_loss = 0.5 * torch.square(pred_values - returns)

            value_error = torch.square(pred_values.detach() - returns.detach())

            # apply loss mask
            assert entropy.shape == pg_loss.shape == value_loss.shape == mask.shape
            entropy = masked_mean(entropy, mask.detach())
            pg_loss = masked_mean(pg_loss, mask.detach())
            value_loss = masked_mean(value_loss, mask.detach())
            value_error = masked_mean(value_error, mask.detach())

            entropy_loss = self.entropy_coef * entropy
            value_loss = self.value_coef * value_loss

            # Negative sign to indicate we want to maximize the policy gradient objective function and entropy
            loss = -(pg_loss + entropy_loss) + value_loss
            scaled_loss = loss * self.loss_scale  # scale the loss to account for gradient accumulation
            scaled_loss.backward()

            # logging only
            stats['pg_loss'].append(pg_loss.detach().item())
            stats['entropy'].append(entropy.detach().item())
            if self.entropy_coef > 0:
                stats['entropy_loss'].append(entropy_loss.detach().item())
            stats['value_loss'].append(value_loss.detach().item())
            stats['value_error'].append(value_error.detach().item())

        # Compute ptx loss, mixing pre-training gradients into PPO,
        # we do it separately because a single backward() call will cause CUDA OOM
        if self.ptx_coef > 0:
            ptx_losses = self.compute_ptx_pretrain_loss()
            stats['ptx_loss'] = ptx_losses

        grad_norm = get_grad_norm_local(self.policy_model)

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.grad_clip,
                error_if_nonfinite=True,
            )

        self.policy_optimizer.step()
        self.policy_scheduler.step()

        t1 = time.time()

        self.c_policy_update += 1
        # Average over accumulated steps
        stats = {k: np.mean(v) for k, v in stats.items() if len(v) > 0}
        stats['learning_rate'] = self.policy_optimizer.param_groups[0]['lr']
        stats['grad_norm'] = grad_norm.item()
        stats['step_time'] = t1 - t0
        return stats

    def log_selfplay_episode_stats(
        self,
        epoch: int,
        episodes: List[Mapping[Text, torch.Tensor]],
        selfplay_sample_interval: int,
        is_training: bool = False,
    ) -> Tuple[List[Mapping[Text, Any]], Mapping[Text, Any]]:
        tb_prefix = 'train_episodes' if is_training else 'val_episodes'

        stats_list = []
        for episode in episodes:
            mask = episode['mask']
            stats = {
                'episode_steps': (episode['terminal_step'] - episode['start_step']).item(),
                'scalar_reward': episode['scalar_reward'].item(),
                'rewards_with_penalties': torch.sum(episode['rewards'] * mask.float()).item(),
                'kl': torch.sum(episode['kl'] * mask.float()).item(),
                'kl_penalties': torch.sum(episode['kl_penalties'] * mask.float()).item(),
                'kl_coef': episode['kl_coef'],
                'episode_time': episode['episode_time'],
            }

            if self.apply_truncate_penalty:
                stats['truncate_penalties'] = torch.sum(episode['truncate_penalties'] * mask.float()).item()

            stats_list.append(stats)

            if is_training:
                self.c_train_episode += 1
                episode_count = self.c_train_episode
            else:
                self.c_val_episode += 1
                episode_count = self.c_val_episode

            if self.tb_writer:
                if episode_count % self.selfplay_log_interval == 0:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            self.tb_writer.add_scalar(f'{tb_prefix}/{k}', v, episode_count)

                # sample generated text
                if episode_count % selfplay_sample_interval == 0:
                    prompt_tokens = episode['tokens'][: episode['start_step'] + 1].tolist()
                    response_tokens = episode['tokens'][episode['start_step'] + 1 : episode['terminal_step'] + 1].tolist()
                    prompt_text = self.tokenizer.decode(prompt_tokens)
                    response_text = self.tokenizer.decode(response_tokens)

                    # Works best when enable 'MarkDown' in tensorboard
                    self.tb_writer.add_text(
                        f'{tb_prefix}_sample',
                        f"**Prompt**: {prompt_text}   <br><br>**Response**: {response_text}   <br><br>**Reward**: {stats['scalar_reward']:.2f}",
                        episode_count,
                    )

        # aggregate over epoch
        aggr_keys = stats_list[0].keys()
        epoch_stats = {}
        for k in aggr_keys:
            epoch_stats[k] = np.mean([d[k] for d in stats_list])

        epoch_tb_prefix = 'train_epochs' if is_training else 'val_epochs'
        if self.tb_writer:
            for k, v in epoch_stats.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f'{epoch_tb_prefix}/{k}', v, epoch)

    def log_train_stats(self, stats: Dict[Text, Any]) -> None:
        tb_prefix = 'ppo_policy'
        if self.tb_writer is not None:
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f'{tb_prefix}/{k}', v, self.c_policy_update)


def main():
    assert cfg.train_log_interval >= 1
    assert cfg.selfplay_log_interval >= 1
    assert cfg.gradient_accum_steps >= 1
    assert 0 < cfg.loss_scale <= 1
    assert cfg.min_gen_len >= 1
    assert cfg.train_batch_size >= 1
    assert cfg.ckpt_interval >= 1
    assert cfg.rollout_size >= (cfg.gradient_accum_steps * cfg.train_batch_size)

    batch_size = int(cfg.train_batch_size * cfg.gradient_accum_steps)

    if not os.path.exists(cfg.sft_ckpt_file):
        raise ValueError(f'Invalid SFT model checkpoint {cfg.sft_ckpt_file!r}, aborting ...')
    if not os.path.exists(cfg.rm_ckpt_file):
        raise ValueError(f'Invalid RM model checkpoint {cfg.rm_ckpt_file!r}, aborting ...')
    if not os.path.exists(cfg.policy_ckpt_file):
        raise ValueError(f'Invalid policy model checkpoint {cfg.policy_ckpt_file!r}, aborting ...')
    if not (torch.version.cuda and torch.cuda.is_bf16_supported()):
        raise RuntimeError('The script only supports training using CUDA and torch.bfloat16, but GPU does not support it.')
    if not any(['cuda' in k for k in (cfg.env_device, cfg.policy_device)]):
        raise ValueError('Bitsandbytes 4bit quantization only works with CUDA, aborting ...')
    if cfg.ptx_coef > 0:
        assert cfg.train_ptx_datasources is not None

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

    train_ptx_loader = None
    if cfg.train_ptx_datasources is not None:
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
        model_cfg=cfg,
        vocab_size=vocab_size,
        ckpt_file=cfg.sft_ckpt_file,
        device=cfg.env_device,
        compute_dtype=compute_dtype,
        frozen=True,
        model_type=cfg.policy_model_type,
        head_type='lm_head',
    )
    reward_model = build_model(
        model_cfg=cfg,
        vocab_size=vocab_size,
        ckpt_file=cfg.rm_ckpt_file,
        device=cfg.env_device,
        compute_dtype=compute_dtype,
        frozen=True,
        model_type=cfg.reward_model_type,
        head_type='scalar_head',
    )

    clip_reward_fn = partial(clip_reward, max_abs_reward=cfg.clip_env_reward)
    train_env = PromptEnv(
        prompt_dataset=train_prompt_dataset,
        reward_model=reward_model,
        sft_model=sft_model,
        normalize_reward=cfg.normalize_env_rewards,
        normalizer_ckpt=cfg.rm_normalizer_ckpt_file,
        clip_reward_fn=clip_reward_fn,
        device=cfg.env_device,
    )
    eval_env = PromptEnv(
        prompt_dataset=val_prompt_dataset,
        reward_model=reward_model,
        sft_model=sft_model,
        normalize_reward=cfg.normalize_env_rewards,
        normalizer_ckpt=cfg.rm_normalizer_ckpt_file,
        clip_reward_fn=clip_reward_fn,
        device=cfg.env_device,
    )

    logger.info('Initializing PPO policy and value model ...')

    # Load model checkpoint using strict=False,
    # because there are missing keys due to LoRA weights and the value head weights not contained in checkpoint state
    policy_model = build_model(
        model_cfg=cfg,
        vocab_size=vocab_size,
        ckpt_file=cfg.policy_ckpt_file,
        device=cfg.policy_device,
        compute_dtype=compute_dtype,
        frozen=False,
        model_type=cfg.policy_model_type,
        head_type='dual_head',  # policy model with an additional value head
        strict=False,
    )

    # initialize the value scalar head weights from RM model, as our fine-tuned checkpoint does not have scalar head
    if os.path.exists(cfg.rm_ckpt_file):
        logger.info(f'Initializing PPO policy value head from checkpoint {cfg.rm_ckpt_file!r} ...')
        ckpt_state = torch.load(cfg.rm_ckpt_file)

        # Make sure don't set other weights accidentally
        value_head_state = {k: v for k, v in ckpt_state.items() if 'scalar_head' in k}
        assert all(['scalar_head' in k for k in value_head_state.keys()])
        policy_model.load_state_dict(value_head_state, strict=False)
        del ckpt_state, value_head_state

    mark_only_lora_as_trainable(policy_model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    max_train_steps = int(cfg.ppo_update_epochs * (cfg.max_episodes / batch_size))
    num_epochs = cfg.max_episodes // cfg.rollout_size

    policy_optimizer = create_optimizer(
        model=policy_model,
        lr=cfg.init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        paged_adamw=cfg.use_paged_adamw,
    )

    policy_scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=policy_optimizer,
        init_lr=cfg.init_lr,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.lr_warmup_steps,
        max_decay_steps=max_train_steps,
    )

    ppo_agent = PPOAgent(
        tokenizer=tokenizer,
        ptx_loader=train_ptx_loader,
        policy_model=policy_model,
        policy_optimizer=policy_optimizer,
        policy_scheduler=policy_scheduler,
        ppo_update_epochs=cfg.ppo_update_epochs,
        train_batch_size=cfg.train_batch_size,
        gradient_accum_steps=cfg.gradient_accum_steps,
        loss_scale=cfg.loss_scale,
        policy_clip_eps=cfg.policy_clip_eps,
        value_clip_eps=cfg.value_clip_eps,
        kl_ctl=AdaptiveKLController(
            init_kl_coef=cfg.init_kl_coef, adaptive=cfg.adaptive_kl, target=cfg.kl_target, horizon=cfg.adaptive_kl_horizon
        ),
        truncate_token=cfg.truncate_token,
        truncate_penalty_value=cfg.truncate_penalty_value,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        ptx_coef=cfg.ptx_coef,
        discount=cfg.discount,
        gae_lambda=cfg.gae_lambda,
        grad_clip=cfg.grad_clip,
        selfplay_log_interval=cfg.selfplay_log_interval,
        train_log_interval=cfg.train_log_interval,
        policy_device=cfg.policy_device,
        tb_writer=SummaryWriter(os.path.join(cfg.log_dir, cfg.policy_model_type)),
    )

    # --------------- Start Training ---------------

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):  # one epoch is just M episodes
        logger.info(f'Epoch {epoch}')
        logger.info(f'Starting to generate {cfg.rollout_size} train episodes ...')

        train_episodes = ppo_agent.run_selfplay(
            epoch=epoch,
            env=train_env,
            num_episodes=cfg.rollout_size,
            batch_size=cfg.selfplay_batch_size,
            temperature=cfg.train_temperature,
            top_p=cfg.train_top_p,
            min_gen_len=cfg.min_gen_len,
            max_gen_len=cfg.max_seq_len,
            selfplay_sample_interval=cfg.selfplay_sample_interval,
            is_training=True,
        )

        # poor solution to swap model between devices, so the training can run without CUDA OOM
        if cfg.env_device != 'cpu' and cfg.env_device == cfg.policy_device:
            reward_model.to('cpu')
            sft_model.to('cpu')

        logger.info(f'Starting to train the agent using {len(train_episodes)} selfplay episodes ...')

        torch.cuda.empty_cache()
        ppo_agent.run_ppo_training_steps(
            episodes=train_episodes,
            whiten_rewards=cfg.whiten_rewards,
            whiten_advantages=cfg.whiten_advantages,
        )

        if cfg.env_device != 'cpu' and cfg.env_device == cfg.policy_device:
            reward_model.to(cfg.env_device)
            sft_model.to(cfg.env_device)

        # regular checkpointing
        if epoch % cfg.ckpt_interval == 0:
            create_lora_checkpoint(
                policy_model,
                os.path.join(cfg.ckpt_dir, f'lora_{cfg.policy_model_type}-epoch-{epoch}.pth'),
                cfg.train_bias,
                cfg.train_head,
            )

        # validation episodes
        if cfg.var_interval > 0 and epoch % cfg.var_interval == 0:
            logger.info(f'Starting to generate {cfg.val_episodes_per_epoch} validation episodes ...')
            _ = ppo_agent.run_selfplay(
                epoch=epoch,
                env=eval_env,
                num_episodes=cfg.val_episodes_per_epoch,
                batch_size=cfg.selfplay_batch_size,
                temperature=cfg.val_temperature,
                top_p=cfg.val_top_p,
                min_gen_len=cfg.min_gen_len,
                max_gen_len=cfg.max_seq_len,
                selfplay_sample_interval=max(10, cfg.selfplay_sample_interval * 0.1),
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
