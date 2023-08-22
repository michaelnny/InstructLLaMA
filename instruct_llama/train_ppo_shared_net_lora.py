"""Train model using PPO algorithm (RL), starting from fine-tuned model and reward model (RM) checkpoints, and with LoRA parameter efficient method."""

import os
import itertools
import functools
from typing import Tuple, List, Mapping, Text, Any
import tqdm
import random
import math
import logging
import time
import copy
import numpy as np
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

import torch.distributed as dist


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer, ModelArgs
from instruct_llama.tokenizer import Tokenizer
from instruct_llama.utils import BlendedDataset, PromptOnlyDataset
from instruct_llama.lora import lora, lora_state_dict, mark_only_lora_as_trainable

from instruct_llama.generation import sample_top_p
from instruct_llama.configs.train_ppo_lora import config as cfg


from instruct_llama.utils import (
    LinearWarmupLRScheduler,
    create_logger,
)


def setup():
    # initialize the process group
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f'clearing cache for rank {rank}')
    torch.cuda.empty_cache()


def create_trace_profiler(tb_trace_dir):
    torch_profiler = torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
        profile_memory=True,
        with_stack=False,
        record_shapes=False,
    )

    return torch_profiler


def create_optimizer(
    model: torch.nn.Module, lr: float, eps: float, weight_decay: float, betas: Tuple[float], fused: bool
) -> torch.optim.AdamW:
    """
    Returns the PyTorch AdamW optimizer for the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    # filter out those do not require gradients
    params_dict = {p_name: params for p_name, params in model.named_parameters() if params.requires_grad}

    # Create empty lists to store parameters for weight decay and no weight decay.
    decay = []
    no_decay = []

    for p_name, params in params_dict.items():
        # Check for parameters corresponding to torch.nn.LayerNorm or torch.nn.Embedding.
        # Note we use hard-coded names where 'ln' is for LayerNorm, and 'embed' is for Embedding, this works better with FSDP
        if (
            p_name.endswith('bias')
            or p_name.endswith('attention_norm.weight')
            or p_name.endswith('ffn_norm.weight')
            or p_name.endswith('post_norm.weight')
            or p_name.endswith('token_embeddings.weight')
        ):
            no_decay.append(params)
        else:
            decay.append(params)

    num_decay_params = sum(p.numel() for p in decay)
    num_nodecay_params = sum(p.numel() for p in no_decay)
    total_num_params = sum(p.numel() for p in params_dict.values())
    assert num_decay_params + num_nodecay_params == total_num_params

    print(f'num decayed parameter tensors: {len(decay)}, with {num_decay_params:,} parameters')
    print(f'num non-decayed parameter tensors: {len(no_decay)}, with {num_nodecay_params:,} parameters')

    # create the pytorch optimizer object
    optim_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    if cfg.use_bnb_8bit:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=lr, eps=eps, betas=betas)
    else:
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, eps=eps, betas=betas, fused=fused)

    return optimizer


# adapted from
# https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/utils/core.py#L336e
def whiten(values, shift_mean=True):
    mean = torch.mean(values, dim=list(range(values.dim())))
    var = torch.var(values, dim=list(range(values.dim())), unbiased=False)  # PyTorch uses biased variance by default

    whitened = (values - mean) * torch.rsqrt(var + 1e-8)

    if not shift_mean:
        whitened += mean
    return whitened


def split_indices_into_bins(bin_size: int, max_indices: int, min_indices: int = 0, shuffle: bool = False) -> List[List[int]]:
    """Split indices to small bins."""

    bin_size = int(bin_size)
    max_indices = int(max_indices)
    min_indices = int(min_indices)

    if max_indices < bin_size:
        raise ValueError(f'Expect max_indices to be greater than bin_size, got {max_indices} and {bin_size}')

    # Split indices into 'bins' with bin_size.
    indices = np.arange(min_indices, max_indices)

    if shuffle:
        np.random.shuffle(indices)

    indices_list = []
    for i in range(0, len(indices), bin_size):
        indices_list.append(indices[i : i + bin_size])  # noqa: E203

    # Make sure the last one has the same 'bin_size'.
    if len(indices_list[-1]) != bin_size:
        indices_list[-1] = indices[-bin_size:]

    return indices_list


class PPOAgent:
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.AdamW,
        scheduler: LinearWarmupLRScheduler,
        reward_model: Transformer,
        sft_model: Transformer,
        tokenizer: Tokenizer,
        prompt_dataset: PromptOnlyDataset,
        pretrain_loader: DataLoader,
        device: torch.device,
        torch_profiler: torch.profiler.profile,
        tb_writer: SummaryWriter,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # the reward model and SFT reference model are fixed
        self.reward_model = reward_model.eval()
        self.sft_model = sft_model.eval()
        self.tokenizer = tokenizer
        self.prompt_dataset = prompt_dataset
        self.pretrain_loader = pretrain_loader
        self.model_params = model.params

        self.discount = cfg.discount
        self.gae_lambda = cfg.gae_lambda
        self.clip_eps = cfg.clip_eps
        self.value_clip_eps = cfg.value_clip_eps

        self.kl_coef = cfg.kl_coef
        self.ptx_coef = cfg.ptx_coef
        self.value_coef = cfg.value_coef
        self.entropy_coef = cfg.entropy_coef
        self.grad_clip = cfg.grad_clip

        self.normalize_reward = cfg.normalize_reward
        self.normalize_advantage = cfg.normalize_advantage

        self.update_epochs = cfg.update_epochs
        self.gradient_accum_steps = cfg.gradient_accum_steps
        self.micro_batch_size = cfg.micro_batch_size

        self.train_log_interval = cfg.train_log_interval
        self.selfplay_log_interval = cfg.selfplay_log_interval
        self.ckpt_interval = cfg.ckpt_interval

        self.torch_profiler = torch_profiler
        self.tb_writer = tb_writer
        self.logger = logger

        self.device = device

        self.batched_episodes = []

        self.c_epidoes = 0
        self.c_updates = 0

    def compute_returns_and_advantages(self, v_t, r_t, v_tp1, done_tp1):
        if self.normalize_reward:
            r_t = whiten(r_t, shift_mean=False)

        discount_tp1 = (~done_tp1).float() * self.discount

        lambda_ = torch.ones_like(discount_tp1) * self.gae_lambda  # If scalar, make into vector.

        delta_t = r_t + discount_tp1 * v_tp1 - v_t

        advantage_t = torch.zeros_like(delta_t, dtype=torch.float32)

        gae_t = 0
        for i in reversed(range(len(delta_t))):
            gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
            advantage_t[i] = gae_t

        return_t = advantage_t + v_t

        if self.normalize_advantage:
            advantage_t = whiten(advantage_t)

        return return_t, advantage_t

    def do_learning(self):
        # always disable cache for learning
        self.model.disable_cache()
        self.model = self.model.train()

        transitions = self.get_transitions_from_episodes()
        del self.batched_episodes[:]

        torch.cuda.empty_cache()
        self.model = self.model.cuda()

        # due to limited GPU resources, we use too loops because we want to avoid constantly moving models to CPU and GPU
        # Run M epochs to update policy model parameters
        for _ in range(self.update_epochs):
            # Split transitions into micro batches
            batch_indices = split_indices_into_bins(self.micro_batch_size, len(transitions), shuffle=True)
            micro_batches = [self.get_batched_transitions([transitions[i] for i in indices]) for indices in batch_indices]

            for i in range(0, len(micro_batches), self.gradient_accum_steps):
                self.update_model(micro_batches[i : i + self.gradient_accum_steps])

        # Remove old transitions
        del transitions

    @torch.no_grad()
    def compute_sft_kl_penalties_for_episodes(self, batched_episodes):
        self.sft_model = self.sft_model.cuda()
        torch.cuda.empty_cache()

        processed_episodes = []
        for episodes in batched_episodes:
            tokens = episodes['padded_safe_tokens'].cuda()
            actions = episodes['padded_actions'].cuda()
            logprobs = episodes['padded_logprobs'].cuda()
            sft_logits = self.sft_model(tokens)  # [batch_size, seq_len, vocab_size]
            sft_dist = Categorical(logits=sft_logits)
            sft_logprobs = sft_dist.log_prob(actions)
            kl_penalties = -self.kl_coef * (logprobs - sft_logprobs)
            episodes['padded_kl_penalties'] = kl_penalties.cpu()
            processed_episodes.append(episodes)

        self.sft_model = self.sft_model.cpu()
        return processed_episodes

    @torch.no_grad()
    def compute_rewards_for_episodes(self, batched_episodes):
        self.reward_model = self.reward_model.cuda()
        torch.cuda.empty_cache()

        processed_episodes = []
        for episodes in batched_episodes:
            tokens = episodes['padded_safe_tokens'].cuda()
            rm_outputs = self.reward_model(tokens).squeeze(-1)  # [batch_size, seq_len]
            episodes['padded_rm_outputs'] = rm_outputs.cpu()
            processed_episodes.append(episodes)

        self.reward_model = self.reward_model.cpu()
        return processed_episodes

    def get_transitions_from_episodes(self):
        # all these mess because lack of GPU resources, as we can't compute them during selfplay using a single GPU
        self.model = self.model.cpu()
        self.logger.info('Starting to compute KL penalties for episodes...')
        # need to add pre-token KL penalties to the reward
        episodes_with_kl_penalties = self.compute_sft_kl_penalties_for_episodes(self.batched_episodes)
        self.logger.info('Starting to compute rewards for episodes...')
        # need to get the environment reward from the reward model
        episodes_with_rm_outputs = self.compute_rewards_for_episodes(episodes_with_kl_penalties)
        self.logger.info('Starting to clean up episodes...')
        cleaned_episodes = self.clean_up_episodes(episodes_with_rm_outputs)
        self.model = self.model.cuda()

        # part of regular PPO algorithm
        # for each episode, compute the returns and advantages before we use these to train the models
        transitions = []
        for episode in cleaned_episodes:
            s_t = episode['states']
            a_t = episode['actions']
            logprob_a_t = episode['logprobs']

            # compute returns and GAE advantages
            v_t = episode['state_values']
            r_t = episode['rewards']

            # pad 0 to make sure v_tp1 and v_t have same length, so can work with compute_returns_and_advantages
            v_tp1 = torch.zeros_like(v_t)
            done_tp1 = torch.ones_like(v_t, dtype=torch.bool)

            v_tp1[:-1] = episode['state_values'][1:]
            done_tp1[:-1] = episode['dones'][1:]

            return_t, advantage_t = self.compute_returns_and_advantages(v_t, r_t, v_tp1, done_tp1)

            transitions.append(
                {
                    's_t': s_t,
                    'a_t': a_t,
                    'logprob_a_t': logprob_a_t,
                    'return_t': return_t,
                    'advantage_t': advantage_t,
                }
            )

        return transitions

    def compute_pretrain_loss(self):
        x, y = next(iter(self.pretrain_loader))
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        y_pred = self.policy_model(x)
        ptx_loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), reduction='mean')
        return ptx_loss

    def update_model(self, micro_batches: List[Mapping[Text, torch.Tensor]]) -> None:
        self.optimizer.zero_grad()

        stats = {
            'pg_loss': 0,
            'entropy': 0,
            'ptx_loss': 0,
            'value_loss': 0,
            'total_loss': 0,
        }

        # accumulate gradients over N micro batches
        for batch in micro_batches:
            s_t = (batch['s_t']).to(device=self.device, dtype=torch.long)
            a_t = (batch['a_t']).to(device=self.device, dtype=torch.long)  # Actions are discrete
            behavior_logprob_a_t = (batch['logprob_a_t']).to(device=self.device, dtype=torch.float)
            return_t = (batch['return_t']).to(device=self.device, dtype=torch.float)
            advantage_t = (batch['advantage_t']).to(device=self.device, dtype=torch.float)
            mask = (batch['mask']).to(device=self.device, dtype=torch.long)

            # Given past states, get predicted action probabilities and state value
            outputs = self.model(s_t)
            pi_logits_t, v_t = outputs['logits'], outputs['values'].squeeze(-1)

            pi_m = Categorical(logits=pi_logits_t)
            pi_logprob_a_t = pi_m.log_prob(a_t)
            entropy = pi_m.entropy()

            assert len(pi_logprob_a_t) == len(behavior_logprob_a_t)
            assert len(v_t) == len(return_t)

            # Compute clipped surrogate objective
            ratio = torch.exp(pi_logprob_a_t - behavior_logprob_a_t.detach())
            clipped_ratio = torch.clamp(ratio, min=1.0 - self.clip_eps, max=1.0 + self.clip_eps)
            pg_loss = torch.min(ratio * advantage_t.detach(), clipped_ratio * advantage_t.detach())
            pg_loss *= mask

            # Compute state value loss, where we use a non-standard method other than traditional PPO or Actor-Critic algorithm
            # here we also clip the value ranges similar to the policy objective
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L343C38-L347
            v_t_clipped = torch.clamp(v_t, return_t - self.value_clip_eps, return_t + self.value_clip_eps)
            vloss_1 = torch.square(v_t - return_t)
            vloss_2 = torch.square(v_t_clipped - return_t)
            value_loss = 0.5 * torch.maximum(vloss_1, vloss_2)
            value_loss *= mask

            # Averaging over batch and time dimension
            pg_loss = torch.mean(pg_loss)
            entropy = torch.mean(entropy)
            value_loss = torch.mean(value_loss)

            # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
            loss = -(pg_loss + self.entropy_coef * entropy) + self.value_coef * value_loss
            scaled_loss = loss / len(micro_batches)
            scaled_loss.backward()

            # Compute ptx loss, mixing pretraining gradients into PPO
            del s_t, a_t, behavior_logprob_a_t, return_t, advantage_t, mask
            ptx_loss = self.compute_pretrain_loss()
            ptx_loss = self.ptx_coef * torch.mean(ptx_loss)
            scaled_ptx_loss = ptx_loss / len(micro_batches)
            scaled_ptx_loss.backward()

            stats['pg_loss'] += pg_loss.detach().item()
            stats['entropy'] += entropy.detach().item()
            stats['value_loss'] += value_loss.detach().item()
            stats['ptx_loss'] += ptx_loss.detach().item()
            stats['total_loss'] += (loss + ptx_loss).detach().item()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                max_norm=self.grad_clip,
                error_if_nonfinite=True,
            )

        self.optimizer.step()
        self.scheduler.step()

        self.c_updates += 1

        if self.c_updates % self.train_log_interval == 0:
            stats = {k: v / len(micro_batches) for k, v in stats.items()}
            stats['lr'] = self.optimizer.param_groups[0]['lr']
            self.log_train_stats(stats)

        if self.c_updates % self.ckpt_interval == 0:
            self.save_model_ckpt()

    def save_model_ckpt(self):
        # save model state
        checkpoint = lora_state_dict(self.model, train_bias=cfg.train_bias, train_head=cfg.train_head)
        torch.save(checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}_train-{self.c_policy_updates}.pth'))

    def run_some_episodes(self, num_episodes: int, batch_size: int, temperature: float, top_p: float):
        # always use cache to speed up acting
        self.model.enable_cache()
        self.model = self.model.eval()
        self.model = self.model.cuda()
        torch.cuda.empty_cache()

        for _ in range(num_episodes // batch_size):
            self.run_batched_episodes(batch_size, temperature, top_p)

    @torch.no_grad()
    def run_batched_episodes(self, batch_size: int, temperature: float, top_p: float):
        assert batch_size >= 1
        t0 = time.time()

        params = self.model_params
        max_gen_len = params.max_seq_len - 1

        # randomly sample a prompt to start a new episode
        prompt_tokens = self.prompt_dataset.sample(batch_size)

        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        input_text_mask = tokens != pad_id

        # temporarily store the data, later will clean it up
        actions = torch.zeros((bsz, total_len), dtype=torch.long)
        actions_logprobs = torch.zeros((bsz, total_len), dtype=torch.float)
        state_values = torch.zeros((bsz, total_len), dtype=torch.float)

        # RL agent starts selfplay
        for cur_pos in range(min_prompt_len, total_len):
            output = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits = output['logits']  # [batch_size, seq_length, vocab_size]
            values = output['values'].squeeze(-1)  # [batch_size, seq_length]

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            # store chosen action, action log probability, and the state values
            # it's ok some of these are incorrectly aligned with the actual completion tokens,
            # as we'll be using a mask to ignore those corresponding to the prompt sequence
            pi_dist = Categorical(logits=logits[:, -1])
            actions[:, cur_pos] = next_token.cpu()
            actions_logprobs[:, cur_pos] = pi_dist.log_prob(next_token).cpu()
            state_values[:, cur_pos] = values[:, -1].cpu()

            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        t1 = time.time()

        # compute a loss mask so we can ignore those incorrectly aligned prompt tokens and also ignore padding tokens
        # We first get a padding mask, then take element-wise OR operation between the input mask and padding mask, then take the inverse
        # the final completion_mask has True for the completion tokens, and False otherwise (prompt or padding tokens)
        padding_mask = tokens == pad_id
        completion_mask = ~(input_text_mask | padding_mask)

        # cut to EOS if any
        # Create a boolean mask where the tensor is equal to the value to find
        completion_tokens = torch.where(input_text_mask, 0, tokens)
        eos_mask = completion_tokens == self.tokenizer.eos_id

        # Find the indices of the first occurrence of the value in each row
        eos_indices = torch.argmax(eos_mask.float(), dim=1)

        for i, eos_idx in enumerate(eos_indices.tolist()):
            if eos_idx == 0:  # skip if there's no EOS token
                continue
            completion_mask[i, :eos_idx] = False

        def get_last_true_indices(mask: torch.Tensor) -> torch.Tensor:
            """Given a 2D boolean mask, returns a 1D vector contains the indices indicating the last True values of each row."""
            reversed_mask = torch.flip(mask.float(), [1])
            # Find the indices of the first True value along each row (in the reversed mask)
            first_true_indices = torch.argmax(reversed_mask, dim=1)
            # Calculate the indices of the last True value in the original mask
            last_true_indices = mask.shape[1] - 1 - first_true_indices
            return last_true_indices

        # get the terminal step for each episode, which is required to compute the reward later in the process
        terminal_steps = get_last_true_indices(completion_mask)

        # replace pad token id (-1) with eos id, as model can't handle -1 as input to the embedding layers
        safe_tokens = torch.where(tokens == pad_id, self.tokenizer.eos_id, tokens)
        self.batched_episodes.append(
            {
                'prompt_tokens': prompt_tokens,
                'padded_safe_tokens': safe_tokens.cpu(),
                'padded_tokens': tokens.cpu(),
                'padded_actions': actions.cpu(),
                'padded_logprobs': actions_logprobs.cpu(),
                'padded_state_values': state_values.cpu(),
                'mask': completion_mask.cpu(),
                'terminal_steps': terminal_steps.cpu(),
            }
        )

        self.logger.info(f'Agent finished generating {batch_size} episodes in {t1 - t0:.2f} seconds')

        return

        # TODO:
        # remove this to the place where we prepare the list of episodes before do learning
        # also add state values since we're using separate models

        def compute_rewards_and_kl_penalties(tokens, actions, logprobs):
            # place the model to CPU so we can re-use the single GPU to speed up things
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

            # replace invalid pad id (-1) before use as input
            input_tokens = torch.where(tokens == pad_id, self.tokenizer.eos_id, tokens)
            input_tokens = input_tokens.cuda()

            # compute the rewards, note it includes values we don't actually need
            # will clean it up later
            self.reward_model = self.reward_model.cuda()
            rm_outputs = self.reward_model(input_tokens).squeeze(-1)  # [batch_size, sequence_length]
            self.reward_model = self.reward_model.cpu()

            # compute pre-token KL penalties
            self.sft_model = self.sft_model.cuda()
            sft_outputs = self.sft_model.forward(input_tokens)
            self.sft_model = self.sft_model.cpu()

            actions = actions.cuda()
            logprobs = logprobs.cuda()

            sft_dist = Categorical(logits=sft_outputs)
            sft_logprobs = sft_dist.log_prob(actions)
            assert sft_logprobs.shape == logprobs.shape

            # here we use log trick, where log(A/B) = log(A) - log(B)
            kl = logprobs - sft_logprobs
            kl_penalties = -self.kl_coef * kl

            # move model back to GPU
            torch.cuda.empty_cache()
            self.model = self.model.cuda()

            return rm_outputs.cpu(), kl_penalties.cpu()

        t1 = time.time()
        rm_outputs, kl_penalties = compute_rewards_and_kl_penalties(tokens, actions, actions_logprobs)
        t2 = time.time()

        t_act = t1 - t0
        t_reward = t2 - t1
        t_total = t2 - t0

        # build the final transitions need to train the policy using PPO algorithm
        actions_logprobs = actions_logprobs.tolist()
        state_values = state_values.tolist()
        rm_outputs = rm_outputs.tolist()
        kl_penalties = kl_penalties.tolist()

        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = len(prompt_tokens[i])
            end = len(prompt_tokens[i]) + max_gen_len

            # we skip the prompt tokens all together
            _actions = toks[start:end]
            _logprobs = actions_logprobs[i][start:end]
            _values = state_values[i][start:end]
            _kl_score = kl_penalties[i][start:end]
            _rm_output = rm_outputs[i][start:end]

            # cut to eos tok if any
            if self.tokenizer.eos_id in _actions:
                eos_idx = _actions.index(self.tokenizer.eos_id)
                _actions = _actions[:eos_idx]
                _logprobs = _logprobs[:eos_idx]
                _values = _values[:eos_idx]
                _kl_score = _kl_score[:eos_idx]
                _rm_output = _rm_output[:eos_idx]

            # skip episode where we don't have enough completion tokens,
            # where minimum is 2, a single word + EOS token
            if len(_actions) < 2:
                continue

            # states are prompt + completion tokens, make room so model can predict the next token
            _states = prompt_tokens[i] + _actions[:-1]

            # rewards are zero for non-terminal states, so we just copy the kl penalties here
            _rewards = copy.deepcopy(_kl_score)
            # get the reward for terminal state, and add it to reward list
            r_T = _rm_output[-1]
            _rewards[-1] += r_T

            _dones = [False] * (len(_values) - 1) + [True]

            self.c_epidoes += 1

            self.episodes.append(
                {
                    'states': torch.tensor(_states, dtype=torch.long),  # environment state
                    'actions': torch.tensor(_actions, dtype=torch.long),  # agent actions (completion tokens)
                    'logprobs': torch.tensor(_logprobs, dtype=torch.float),  # agent action logprobs
                    'state_values': torch.tensor(_values, dtype=torch.float),  # state values
                    'rewards': torch.tensor(_rewards, dtype=torch.float),  # rewards with KL penalties
                    'dones': torch.tensor(_dones, dtype=torch.bool),  # marks for terminal states
                }
            )

            # logging
            if self.c_epidoes % self.selfplay_log_interval == 0:
                stats = {
                    'steps': len(_actions),
                    'agent_act(second)': t_act / len(prompt_tokens),
                    'compute_reward(second)': t_reward / len(prompt_tokens),
                    'total(second)': t_total / len(prompt_tokens),
                    'reward': r_T,
                    'kl_penalties': np.mean(_kl_score).item(),
                }
                self.log_selfplay_stats(stats)

    def log_train_stats(self, stats: Mapping[Text, Any]):
        if self.tb_writer is not None:
            try:
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f'train/{k}', v, self.c_policy_updates)
            except Exception:
                pass

        self.logger.info(f'Training steps: {self.c_policy_updates}')
        self.logger.info(stats)

    def log_selfplay_stats(self, stats: Mapping[Text, Any]):
        if self.tb_writer is not None:
            try:
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f'episode/{k}', v, self.c_epidoes)
            except Exception:
                pass

        if self.c_epidoes % 200 == 0:
            self.logger.info(f'Selfplay episode: {self.c_epidoes}')
            self.logger.info(stats)

    def get_batched_transitions(self, micro_batch: List[Mapping[Text, torch.Tensor]]) -> Mapping[Text, torch.Tensor]:
        batch_size = len(micro_batch)

        max_batch_len = max([len(item['s_t']) for item in micro_batch])

        batch_s_t = torch.full((batch_size, max_batch_len), 0, dtype=torch.long)
        batch_a_t = torch.full((batch_size, max_batch_len), 0, dtype=torch.long)
        batch_logprob_a_t = torch.full((batch_size, max_batch_len), 0, dtype=torch.float)
        batch_return_t = torch.full((batch_size, max_batch_len), 0, dtype=torch.float)
        batch_advantage_t = torch.full((batch_size, max_batch_len), 0, dtype=torch.float)

        mask = torch.full((batch_size, max_batch_len), 0, dtype=torch.int8)

        for i, item in enumerate(micro_batch):
            seq_len = len(item['s_t'])
            action_len = len(item['a_t'])

            start_idx = seq_len - action_len

            batch_s_t[i, :seq_len] = item['s_t']
            batch_a_t[i, start_idx:seq_len] = item['a_t']
            batch_logprob_a_t[i, start_idx:seq_len] = item['logprob_a_t']
            batch_return_t[i, start_idx:seq_len] = item['return_t']
            batch_advantage_t[i, start_idx:seq_len] = item['advantage_t']
            mask[i, start_idx:seq_len] = 1

        return {
            's_t': batch_s_t,
            'a_t': batch_a_t,
            'logprob_a_t': batch_logprob_a_t,
            'return_t': batch_return_t,
            'advantage_t': batch_advantage_t,
            'mask': mask,
        }

    @torch.no_grad()
    def compute_terminal_step_reward(self, tokens: List[int]) -> float:
        tokens = torch.tensor(tokens, dtype=torch.long)

        # [1, sequence_length, 1]
        outputs = self.reward_model(tokens.unsqueeze(0))

        # get reward for terminal time step
        reward = outputs.squeeze()[-1]

        return reward.cpu().item()

    @torch.no_grad()
    def compute_sft_logprobs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        prompt_len = len(states) - len(actions)

        states = states.unsqueeze(0)

        sft_logits = self.sft_model.forward(states)  # [1, seq_len, vocab_size]
        sft_logits = sft_logits.squeeze(0)[prompt_len:]  # [completion_len, vocab_size]
        sft_dist = Categorical(logits=sft_logits)
        ref_logprobs = sft_dist.log_prob(actions)

        return ref_logprobs.cpu()

    def add_kl_penalty(self, rewards, logprobs, ref_logprobs):
        # here we use log trick, where log(A/B) = log(A) - log(B)
        kl = logprobs - ref_logprobs
        kl_penalties = -self.kl_coef * kl
        rewards += kl_penalties
        return rewards, kl_penalties

    @torch.no_grad()
    def run_single_episode(self):
        t0 = time.time()

        # randomly sample a prompt to start a new episode
        prompt_tokens = self.prompt_dataset.sample()

        prompt_len = len(prompt_tokens)
        assert prompt_len <= self.max_seq_len

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((1, self.max_seq_len), pad_id, dtype=torch.long, device=self.device)
        tokens[:, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)

        # store needed transitions for PPO algorithm
        actions = []
        logprobs = []
        state_values = []
        rewards = []
        dones = []

        prev_pos = 0
        for cur_pos in range(prompt_len, self.max_seq_len):
            output = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            logits = output['logits'].squeeze(0)  # [seq_length, vocab_size]
            values = output['values'].squeeze(0).squeeze(-1)  # [seq_length]

            # we only need the last one from the output sequence
            logits = logits[-1]  # [1, vocab_size]
            values = values[-1]  # [1]

            pi_dist = Categorical(logits=logits)

            # assume temperature = 0
            next_token = torch.argmax(logits, dim=-1)  # [1]

            actions.append(next_token.cpu().item())
            logprobs.append(pi_dist.log_prob(next_token).cpu().item())
            state_values.append(values.cpu().item())

            tokens[:, cur_pos] = next_token
            if next_token == self.tokenizer.eos_id:
                break
            elif cur_pos < self.max_seq_len - 1:
                # rewards are always zero for non-terminal state,
                # so we can use standard GAE to compute the returns and advantages
                rewards.append(0.0)
                dones.append(False)

        t1 = time.time()

        # compute the reward for the terminal state
        r_T = self.compute_terminal_step_reward(prompt_tokens + actions)
        rewards.append(r_T)
        dones.append(True)

        t2 = time.time()

        assert len(rewards) == len(state_values)

        # print('-' * 80)
        # print(self.tokenizer.decode(prompt_tokens))
        # print('\n -->')
        # print(self.tokenizer.decode(actions))
        # print('-' * 80)

        # environment states, we skip the final completion token, as we want the agent to choose which action during learning
        states = torch.tensor(prompt_tokens + actions[:-1], dtype=torch.long)
        actions = torch.tensor(actions, dtype=torch.long)  # agent actions
        logprobs = torch.tensor(logprobs, dtype=torch.float32)  # logprobs of agent actions
        state_values = torch.tensor(state_values, dtype=torch.float32)  # predicted state values
        rewards = torch.tensor(rewards, dtype=torch.float32)  # environment reward
        dones = torch.tensor(dones, dtype=torch.bool)  # terminal mark for environment states

        # compute logprobs for completion tokens using the SFT model,
        # which is required to add the KL penalties for pre-token reward
        ref_logprobs = self.compute_sft_logprobs(states, actions)

        t3 = time.time()
        assert len(ref_logprobs) == len(logprobs)

        # add KL penalty to the rewards
        rewards, kl_penalties = self.add_kl_penalty(rewards, logprobs, ref_logprobs)

        self.episodes.append(
            {
                'states': states,
                'actions': actions,
                'logprobs': logprobs,
                'state_values': state_values,
                'rewards': rewards,
                'dones': dones,
            }
        )

        self.c_epidoes += 1

        stats = {
            'steps': len(actions),
            'act_time(second)': t1 - t0,
            'reward_time(second)': t2 - t1,
            'sft_time(second)': t3 - t2,
            'total_time(second)': t3 - t0,
            'reward': r_T,
            'kl_penalties': kl_penalties.mean().item(),
        }

        if self.c_epidoes % self.selfplay_log_interval == 0:
            self.log_selfplay_stats(stats)


def build_model_with_head_type(vocab_size, head_type, ckpt_file, disable_grad, logger) -> Transformer:
    assert vocab_size > 0
    assert head_type in ('lm_head', 'scalar_head', 'lm_and_scalar_heads')

    model_args = ModelArgs.from_model_type(cfg.model_type)
    model_args.vocab_size = vocab_size
    model_args.max_seq_len = cfg.max_seq_len
    model_args.max_batch_size = cfg.selfplay_batch_size
    model_args.use_cache = False
    model_args.embed_dropout = cfg.embed_dropout
    model_args.attn_dropout = cfg.attn_dropout
    model_args.resid_dropout = cfg.resid_dropout
    model_args.head_type = head_type

    model = Transformer(model_args)

    if os.path.exists(ckpt_file):
        logger.info(f'Loading model checkpoint {ckpt_file}...')
        model_state = torch.load(ckpt_file)
        model.load_state_dict(model_state, strict=False)
        del model_state  # free up CPU RAM

    if disable_grad:
        for p in model.parameters():
            p.requires_grad = False

    return model


def trainable_model_to_dtype(model: Transformer):
    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=torch.bfloat16)


def main():
    assert cfg.gradient_accum_steps >= 1
    assert cfg.train_log_interval >= 1
    assert cfg.selfplay_log_interval >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.micro_batch_size >= 1
    assert cfg.learn_interval >= (cfg.gradient_accum_steps * cfg.micro_batch_size)

    if not os.path.exists(cfg.sft_ckpt_file):
        raise ValueError(f'Invalid SFT model checkpoint "{cfg.sft_ckpt_file}", aborting...')
    if not os.path.exists(cfg.rm_ckpt_file):
        raise ValueError(f'Invalid RM model checkpoint "{cfg.rm_ckpt_file}", aborting...')

    if not (torch.version.cuda and torch.cuda.is_bf16_supported()):
        raise RuntimeError('The script only supports training using torch.bfloat16, but GPU does not support it.')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()

    logger = create_logger()

    # # --------------- Load datasets ---------------

    logger.info('Loading datasets...')

    tokenizer = Tokenizer(cfg.tokenizer_file)
    vocab_size = tokenizer.vocab_size

    # Our custom IterableDatasets already have sharding and shuffle mechanism implemented
    cuda_kwargs = {
        'num_workers': cfg.dataloader_workers,
        'batch_size': cfg.micro_batch_size,
        'pin_memory': True,
        'shuffle': False,
        'sampler': None,
    }

    pretrain_dataset = BlendedDataset(
        data_sources=cfg.pretrain_datasources,
        max_seq_len=cfg.max_seq_len,
        rank=rank,
        world_size=world_size,  # shard the dataset
        seed=int(cfg.seed + rank),
    )

    pretrain_loader = DataLoader(pretrain_dataset, **cuda_kwargs)

    logger.info(f'Pretrain dataset metadata:\n{pretrain_dataset.get_metadata()}')

    prompt_dataset = PromptOnlyDataset(
        data_sources=cfg.prompt_datasources,
        max_seq_len=cfg.max_prompt_len,
        max_samples=cfg.max_train_samples,
        seed=cfg.seed,
    )

    logger.info(f'Prompt dataset metadata:\n{prompt_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    logger.info('Initialize model and optimizer...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    logger.info('Initialize SFT model...')
    sft_model = build_model_with_head_type(
        vocab_size=vocab_size, head_type='lm_head', ckpt_file=cfg.sft_ckpt_file, disable_grad=True, logger=logger
    )
    sft_model = sft_model.to(torch.bfloat16)

    logger.info('Initialize RM model...')
    reward_model = build_model_with_head_type(
        vocab_size=vocab_size, head_type='scalar_head', ckpt_file=cfg.rm_ckpt_file, disable_grad=True, logger=logger
    )
    reward_model = reward_model.to(torch.bfloat16)

    logger.info('Initialize model with policy and value heads...')

    with lora(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, enabled=True):
        model = build_model_with_head_type(
            vocab_size=vocab_size,
            head_type='lm_and_scalar_heads',
            ckpt_file=cfg.sft_ckpt_file,
            disable_grad=False,
            logger=logger,
        )

        # only need the scalar_head from RM model checkpoint
        if os.path.exists(cfg.rm_ckpt_file):
            logger.info(f'Loading scalar head weights from RM model checkpoint {cfg.rm_ckpt_file}...')
            rm_state = torch.load(cfg.rm_ckpt_file)
            rm_partial_state = {k: rm_state[k] for k in rm_state.keys() if 'scalar_head' in k}
            model.load_state_dict(rm_partial_state, strict=False)
            del rm_state, rm_partial_state
        else:
            model.init_scalar_head_weights()

    mark_only_lora_as_trainable(model, train_bias=cfg.train_bias, train_head=cfg.train_head)
    trainable_model_to_dtype(model)
    model = model.to(local_rank)

    logger.info('Initialize optimizer...')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
    )

    scheduler = LinearWarmupLRScheduler(
        optimizer=optimizer, init_lr=cfg.init_lr, max_lr=cfg.max_lr, warmup_steps=cfg.warmup_steps
    )

    # --------------- Start Training ---------------

    logger.info(f'Starting to run {cfg.max_episodes} episodes...')

    torch_profiler = None
    tb_writer = None

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        # Careful as the logs will grow very fast
        if cfg.use_profiler:
            torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        reward_model=reward_model,
        sft_model=sft_model,
        tokenizer=tokenizer,
        prompt_dataset=prompt_dataset,
        pretrain_loader=pretrain_loader,
        device=local_rank,
        torch_profiler=torch_profiler,
        tb_writer=tb_writer,
        logger=logger,
    )

    while agent.c_epidoes < cfg.max_episodes:
        # generate training sample episodes

        agent.run_some_episodes(
            num_episodes=cfg.learn_interval,
            batch_size=cfg.selfplay_batch_size,
            temperature=cfg.selfplay_temperature,
            top_p=cfg.selfplay_top_p,
        )

        agent.do_learning()

    if rank == 0:
        # training is done...show some training stats.
        logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
