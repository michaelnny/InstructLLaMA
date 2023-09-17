"""Train model using PPO algorithm (RL), starting from fine-tuned model and reward model (RM) checkpoints, and with LoRA parameter efficient method."""

import os
from typing import Tuple, List, Mapping, Text, Any
import tqdm
import random
import time
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer, RewardModel, ModelArgs
from instruct_llama.configs.train_ppo_lora import config as cfg


from instruct_llama.utils import (
    BlendedDataset,
    PromptOnlyDataset,
    Tokenizer,
    LinearWarmupLRScheduler,
    RunningMeanStd,
    create_optimizer,
    create_logger,
    masked_whiten,
    masked_mean,
    masked_sum,
    get_grad_norm_local,
    split_indices_into_bins,
    sample_top_p,
    lora,
    lora_state_dict,
    mark_only_lora_as_trainable,
)


def setup():
    # initialize the process group
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f'clearing cache for rank {rank}')
    torch.cuda.empty_cache()


# -------------------------------- RL selfplay to generate training samples --------------------------------


@torch.no_grad()
def compute_state_values_for_batched_episodes(
    value_model: RewardModel, batched_episodes: List[Mapping[Text, torch.Tensor]]
) -> List[Mapping[Text, torch.Tensor]]:
    value_model = value_model.cuda()
    torch.cuda.empty_cache()

    processed_episodes = []
    for episodes in batched_episodes:
        tokens = episodes['tokens'].cuda()
        output = value_model(tokens)  # [batch_size, seq_len, 1]
        values = output.squeeze(-1)  # [batch_size, seq_len]
        mask = episodes['mask'].cuda()
        values *= mask.float()

        episodes['values'] = values.cpu()
        processed_episodes.append(episodes)

    value_model = value_model.cpu()
    return processed_episodes


@torch.no_grad()
def compute_kl_penalties_for_batched_episodes(
    sft_model: Transformer, batched_episodes: List[Mapping[Text, torch.Tensor]]
) -> List[Mapping[Text, torch.Tensor]]:
    if cfg.kl_coef > 0:
        sft_model = sft_model.cuda()
        sft_model = sft_model.eval()
        torch.cuda.empty_cache()

    processed_episodes = []
    for episodes in batched_episodes:
        tokens = episodes['tokens'].cuda()

        if cfg.kl_coef > 0:
            sft_logits = sft_model(tokens)  # [batch_size, seq_len, vocab_size]
            del tokens
            actions = episodes['actions'].cuda()
            logprobs = episodes['logprobs'].cuda()
            sft_dist = Categorical(logits=sft_logits)
            sft_logprobs = sft_dist.log_prob(actions)

            mask = episodes['mask'].cuda()
            kl = logprobs - sft_logprobs

            # clip KL
            if cfg.clip_kl > 0:
                kl = torch.clamp(kl, min=-cfg.clip_kl, max=cfg.clip_kl)

            kl = cfg.kl_coef * kl
            kl *= mask.float()
        else:
            kl = torch.zeros_like(tokens)

        episodes['kl'] = kl.cpu()
        processed_episodes.append(episodes)

    if cfg.kl_coef > 0:
        sft_model = sft_model.cpu()
    torch.cuda.empty_cache()
    return processed_episodes


@torch.no_grad()
def compute_env_rewards_for_batched_episodes(
    reward_model: RewardModel,
    batched_episodes: List[Mapping[Text, torch.Tensor]],
    reward_stats: RunningMeanStd,
    update_reward_stats: bool,
) -> List[Mapping[Text, torch.Tensor]]:
    reward_model = reward_model.cuda()
    reward_model = reward_model.eval()
    torch.cuda.empty_cache()

    processed_episodes = []

    for episodes in batched_episodes:
        tokens = episodes['tokens'].cuda()
        terminal_steps = episodes['terminal_steps'].cuda()  # [batch_size]
        # get scalar reward for the terminal step from RM model
        outputs = reward_model(tokens).squeeze(-1)  # [batch_size, seq_length]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        env_reward = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [batch_size]

        if cfg.normalize_env_rewards:
            normed_env_reward = reward_stats.normalize(env_reward, False)
        else:
            normed_env_reward = env_reward

        if update_reward_stats:
            reward_stats.update(env_reward)

        # clip rewards, similar to how we do reward clipping in RL
        if cfg.clip_env_reward > 0:
            normed_env_reward = torch.clamp(normed_env_reward, min=-cfg.clip_env_reward, max=cfg.clip_env_reward)

        # rewards are zero except the terminal step
        rewards_no_kl = torch.zeros_like(tokens, dtype=torch.float)
        for i, idx in enumerate(terminal_steps.tolist()):
            # idx - 1 because the reward is for the last action took by the agent,
            # which leads to the termination of the episode
            rewards_no_kl[i, idx - 1] += normed_env_reward[i]

        episodes['env_reward'] = env_reward.cpu()
        episodes['normed_env_reward'] = normed_env_reward.cpu()
        episodes['reward_no_kl'] = rewards_no_kl.cpu()
        processed_episodes.append(episodes)

    reward_model = reward_model.cpu()
    torch.cuda.empty_cache()
    return processed_episodes


@torch.no_grad()
def add_kl_to_rewards_for_batched_episodes(
    batched_episodes: List[Mapping[Text, torch.Tensor]],
) -> List[Mapping[Text, torch.Tensor]]:
    processed_episodes = []

    for episodes in batched_episodes:
        reward_no_kl = episodes['reward_no_kl'].cuda()
        kl = episodes['kl'].cuda()
        mask = episodes['mask'].cuda()
        rewards = reward_no_kl - kl
        rewards *= mask.float()

        if cfg.whiten_rewards:
            rewards = masked_whiten(rewards, mask, shift_mean=False)

        episodes['rewards'] = rewards.cpu()
        processed_episodes.append(episodes)

    return processed_episodes


@torch.no_grad()
def compute_returns_and_advantages_for_batched_episodes(
    batched_episodes: List[Mapping[Text, torch.Tensor]]
) -> List[Mapping[Text, torch.Tensor]]:
    processed_episodes = []
    for episodes in batched_episodes:
        returns, advantages = compute_masked_returns_and_advantages(episodes)

        episodes['returns'] = returns.cpu()
        episodes['advantages'] = advantages.cpu()
        processed_episodes.append(episodes)

    return processed_episodes


def compute_masked_returns_and_advantages(episodes: Mapping[Text, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    values = episodes['values'].cuda()
    rewards = episodes['rewards'].cuda()
    terminal_steps = episodes['terminal_steps'].cuda()
    mask = episodes['mask'].cuda()

    returns = torch.zeros_like(values, device='cuda')
    advantages = torch.zeros_like(values, device='cuda')

    for i, idx in enumerate(terminal_steps.tolist()):
        r_t = rewards[i, :idx]
        v_t = values[i, :idx]
        v_tp1 = values[i, 1 : idx + 1]

        # diff = len(v_t) - len(v_tp1)
        # if diff > 0:
        #     v_tp1 = F.pad(v_tp1, (0, diff), value=0)

        assert len(r_t) == len(v_t) == len(v_tp1)

        done_tp1 = torch.zeros_like(v_tp1, dtype=torch.bool, device='cuda')
        done_tp1[-1] = True

        discount_tp1 = (~done_tp1).float() * cfg.discount

        adv_t = truncated_generalized_advantage_estimation(r_t, v_t, v_tp1, discount_tp1, cfg.gae_lambda)

        return_t = adv_t + v_t

        returns[i, : len(return_t)] = return_t
        advantages[i, : len(adv_t)] = adv_t

    if cfg.whiten_advantages:
        advantages = masked_whiten(advantages, mask)

    return returns, advantages


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


def batched_episodes_to_transitions(
    batch_episodes: List[Mapping[Text, torch.Tensor]]
) -> Tuple[List[Mapping[Text, torch.Tensor]], List[Mapping[Text, Any]]]:
    """Un-batch the selfplay episodes, since during training, the batch size is much smaller than selfplay.
    We also skip some bad episodes, when the number of completion tokens is lesser than certain threshold.
    """
    skipped = 0
    transitions = []
    stats = []
    for batch in batch_episodes:
        # for each episode in current batch
        for i in range(len(batch['tokens'])):
            # cut to the terminal step
            num_steps = batch['num_steps'][i]
            terminal_step = batch['terminal_steps'][i]

            # skip episode without enough completion tokens
            if num_steps < cfg.min_gen_len:
                skipped += 1
                continue

            end = terminal_step + 1  # plus one because high is exclusive

            # all needed to train the model using PPO algorithm
            transition = {
                'tokens': batch['tokens'][i, :end],
                'actions': batch['actions'][i, :end],
                'logprobs': batch['logprobs'][i, :end],
                'mask': batch['mask'][i, :end],
            }

            # only training episodes have these
            if 'values' in batch:
                transition['values'] = batch['values'][i, :end]
                transition['returns'] = batch['returns'][i, :end]
                transition['advantages'] = batch['advantages'][i, :end]

            transitions.append(transition)

            # logging
            mask = batch['mask'][i, :end]
            kl = batch['kl'][i, :end]
            rewards = batch['rewards'][i, :end]
            env_reward = batch['env_reward'][i]
            normed_env_reward = batch['normed_env_reward'][i]

            stats.append(
                {
                    'steps': num_steps.item(),
                    'env_reward': env_reward.item(),
                    'normed_env_reward': normed_env_reward.item(),
                    'reward': masked_sum(rewards, mask, 0).item(),
                    'kl': masked_sum(kl, mask, 0).item(),
                }
            )

    print(f'Skipped {skipped} episodes with completion tokens lesser than {cfg.min_gen_len}')

    return transitions, stats


@torch.no_grad()
def get_transitions_from_batched_episodes(
    sft_model: Transformer,
    reward_model: RewardModel,
    value_model: RewardModel,
    batched_episodes: List[Mapping[Text, torch.Tensor]],
    reward_stats: RunningMeanStd,
    update_reward_stats: bool,
) -> Tuple[List[Mapping[Text, torch.Tensor]], List[Mapping[Text, Any]]]:
    """Turn the list of batched episodes into list of transitions, so we can use them with the PPO algorithm to train the model"""
    print('Starting to build transitions from selfplay episodes...')

    torch.cuda.empty_cache()

    episodes = compute_kl_penalties_for_batched_episodes(sft_model, batched_episodes)
    episodes = compute_env_rewards_for_batched_episodes(reward_model, episodes, reward_stats, update_reward_stats)
    episodes = add_kl_to_rewards_for_batched_episodes(episodes)

    if value_model is not None:
        episodes = compute_state_values_for_batched_episodes(value_model, episodes)
        episodes = compute_returns_and_advantages_for_batched_episodes(episodes)

    torch.cuda.empty_cache()

    return batched_episodes_to_transitions(episodes)


@torch.no_grad()
def generate_single_batch_episodes(
    policy_model: Transformer,
    tokenizer: Tokenizer,
    prompt_dataset: PromptOnlyDataset,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> Mapping[Text, torch.Tensor]:
    """Run one batch episodes, where the code is adapted from the generation.py module,
    here we also store the intermediate transitions which are required to train the model using the PPO algorithm"""
    assert batch_size >= 4

    params = policy_model.params
    max_gen_len = cfg.max_gen_len
    # randomly sample a batch of prompts
    prompt_tokens = prompt_dataset.sample(batch_size)

    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)

    assert min_prompt_len > 2

    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device='cuda')
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cuda')

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device='cuda')
    input_text_mask = tokens != pad_id

    logprobs = torch.zeros((bsz, total_len), dtype=torch.float, device='cuda')

    # RL agent starts selfplay
    t0 = time.time()
    for cur_pos in range(min_prompt_len, total_len):
        output = policy_model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
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

        # store chosen action, action log probability, and the state values
        # we use cur_pos-1 because cur_pos is actually pointing to next token, not current one
        insert_idx = cur_pos - 1
        pi_dist = Categorical(logits=logits)
        next_token_logprobs = pi_dist.log_prob(next_token)
        next_token_logprobs = torch.where(input_text_mask[:, cur_pos], logprobs[:, insert_idx], next_token_logprobs)
        logprobs[:, insert_idx] = next_token_logprobs

        prev_pos = cur_pos
        if all(eos_reached):
            break

    t1 = time.time()

    start_steps = torch.tensor([len(prompts) for prompts in prompt_tokens], dtype=torch.long, device='cuda')

    terminal_steps = torch.zeros((bsz,), dtype=torch.long, device='cuda')

    # cut tokens to:
    # a. maximum generation length
    # b. eos token
    # c. begin of instruction token [INST]
    def find_begin_of_inst_index(input_list, pattern=[518, 25580, 29962]):
        pattern_length = len(pattern)
        lst_length = len(input_list)
        for i in range(lst_length - pattern_length + 1):
            if input_list[i : i + pattern_length] == pattern:
                return i

        return -1  # Return -1 if pattern is not found

    for i, toks in enumerate(tokens.tolist()):
        start = len(prompt_tokens[i])
        toks = toks[start : start + max_gen_len]

        # cut to max gen len, -1 to avoid out of index
        end_idx = min(start + max_gen_len, total_len) - 1

        # cut to eos token </eos>
        if eos_id in toks:
            end_idx = start + toks.index(eos_id)
        else:
            # cut to begin of instruction token [INST]
            begin_inst_idx = find_begin_of_inst_index(toks)
            if begin_inst_idx > 0:
                end_idx = start + begin_inst_idx

        assert end_idx >= start and end_idx < total_len and end_idx <= start + max_gen_len
        terminal_steps[i] = end_idx

    # count number of steps for each episode
    num_steps = (terminal_steps - start_steps).clamp_min(0)

    # build up loss mask, where we only keep the completions tokens
    # for example, if we have a sequence of:
    # [1, 2, 3, 4, 5, 6, 7, -1, -1]
    # where:
    #   [1, 2, 3, 4] are prompt tokens
    #   [5, 6, 7] are completion tokens
    #   [-1, -1] are padding tokens
    #
    # then the mask will be:
    # [False, False, False, True, True, True, False, False, False]
    mask = torch.zeros_like(tokens, dtype=torch.bool, device='cpu')

    for i, (start_idx, end_idx) in enumerate(zip(start_steps.tolist(), terminal_steps.tolist())):
        mask[i, start_idx - 1 : end_idx] = True
        tokens[i, end_idx:] = eos_id
        logprobs[i, end_idx:] = torch.tensor(0.7).log()

    # shift one step to left to get the actions taken by the agent
    actions = torch.full((bsz, total_len), eos_id, dtype=torch.long, device='cpu')
    actions[:, :-1] = tokens[:, 1:].cpu()

    episodes = {
        'num_steps': num_steps.cpu(),
        'tokens': tokens.cpu(),
        'actions': actions.cpu(),
        'logprobs': logprobs.cpu(),
        'mask': mask.cpu(),
        'start_steps': start_steps.cpu(),
        'terminal_steps': terminal_steps.cpu(),
        'episodes_time': t1 - t0,
    }

    return episodes


@torch.no_grad()
def generate_episodes(
    prompt_dataset: PromptOnlyDataset,
    tokenizer: Tokenizer,
    policy_model: Transformer,
    num_episodes: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> List[Mapping[Text, torch.Tensor]]:
    # always use cache to speed up acting
    policy_model.enable_cache()
    policy_model = policy_model.eval()

    if not next(policy_model.parameters()).is_cuda:
        policy_model = policy_model.cuda()

    torch.cuda.empty_cache()

    # reset counter
    batched_episodes = []

    for _ in range(num_episodes // batch_size):
        batched_episodes.append(
            generate_single_batch_episodes(policy_model, tokenizer, prompt_dataset, batch_size, temperature, top_p)
        )

    policy_model.disable_cache()
    policy_model = policy_model.cpu()
    torch.cuda.empty_cache()

    return batched_episodes


@torch.no_grad()
def run_warmup_episodes(
    prompt_dataset: PromptOnlyDataset,
    tokenizer: Tokenizer,
    policy_model: Transformer,
    reward_model: RewardModel,
    num_episodes: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    reward_stats: RunningMeanStd,
):
    batched_episodes = generate_episodes(prompt_dataset, tokenizer, policy_model, num_episodes, batch_size, temperature, top_p)
    _ = compute_env_rewards_for_batched_episodes(reward_model, batched_episodes, reward_stats, True)


@torch.no_grad()
def run_selfplay_episodes(
    prompt_dataset: PromptOnlyDataset,
    tokenizer: Tokenizer,
    policy_model: Transformer,
    value_model: RewardModel,
    reward_model: RewardModel,
    sft_model: Transformer,
    num_episodes: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    reward_stats: RunningMeanStd,
    update_reward_stats: bool = False,
) -> Tuple[List[Mapping[Text, torch.Tensor]], List[Mapping[Text, Any]], Mapping[Text, Any]]:
    t0 = time.time()
    batched_episodes = generate_episodes(prompt_dataset, tokenizer, policy_model, num_episodes, batch_size, temperature, top_p)

    t1 = time.time()

    act_time = t1 - t0
    avg_act_time = act_time / num_episodes

    print(
        f'Finished generating {num_episodes} episodes in {act_time:.2f} seconds, average time per episode is {avg_act_time:.2f} second'
    )

    t2 = time.time()

    transitions, episode_stats = get_transitions_from_batched_episodes(
        sft_model,
        reward_model,
        value_model,
        batched_episodes,
        reward_stats,
        update_reward_stats,
    )

    t3 = time.time()

    build_time = t3 - t2
    avg_build_time = act_time / len(transitions)

    print(
        f'Finished processing {len(transitions)} episodes in {build_time:.2f} seconds, average time per episode is {avg_build_time:.2f} second'
    )

    env_rewards = torch.tensor([stats['env_reward'] for stats in episode_stats], dtype=torch.float)
    normed_env_rewards = torch.tensor([stats['normed_env_reward'] for stats in episode_stats], dtype=torch.float)
    rewards = torch.tensor([stats['reward'] for stats in episode_stats], dtype=torch.float)
    kl = torch.tensor([stats['kl'] for stats in episode_stats], dtype=torch.float)
    steps = torch.tensor([stats['steps'] for stats in episode_stats], dtype=torch.float)

    epoch_stats = {
        'env_reward_mean': torch.mean(env_rewards).item(),
        'env_reward_std': torch.std(env_rewards).item(),
        'normed_env_reward_mean': torch.mean(normed_env_rewards).item(),
        'normed_env_reward_std': torch.std(normed_env_rewards).item(),
        'reward': torch.mean(rewards).item(),
        'kl': torch.mean(kl).item(),
        'episode_steps': torch.mean(steps).item(),
    }

    return transitions, episode_stats, epoch_stats


# -------------------------------- RL selfplay to generate training samples --------------------------------


# -------------------------------- RL PPO training --------------------------------


def run_ppo_training_steps(
    policy_model: Transformer,
    policy_optimizer: torch.optim.AdamW,
    policy_scheduler: LinearWarmupLRScheduler,
    value_model: RewardModel,
    value_optimizer: torch.optim.AdamW,
    value_scheduler: LinearWarmupLRScheduler,
    transitions: List[Mapping[Text, torch.Tensor]],
    ptx_loader: DataLoader,
    eos_id: int,
) -> Tuple[List[Mapping[Text, Any]], List[Mapping[Text, Any]], List[Mapping[Text, Any]]]:
    # Run M epochs to update policy model parameters
    print(f'Starting to run {cfg.update_epochs} epochs over {len(transitions)} episodes to update policy model...')

    if next(value_model.parameters()).is_cuda:
        value_model = value_model.cpu()

    policy_model = policy_model.train()
    policy_model = policy_model.cuda()
    torch.cuda.empty_cache()

    policy_stats = []
    for _ in range(cfg.update_epochs):
        # Split transitions into micro batches
        batch_indices = split_indices_into_bins(cfg.micro_batch_size, len(transitions), shuffle=True, drop_last=True)
        micro_batches = [get_batched_transitions([transitions[i] for i in indices], eos_id) for indices in batch_indices]

        for i in range(0, len(micro_batches), cfg.gradient_accum_steps):
            stats = update_policy_model(
                micro_batches[i : i + cfg.gradient_accum_steps], policy_model, policy_optimizer, policy_scheduler, ptx_loader
            )
            policy_stats.append(stats)

    # Run M epochs to update state value model parameters, we use two loops
    # because we can't host the two models at the same time with a single GPU
    print(f'Starting to run {cfg.update_epochs} epochs over {len(transitions)} episodes to update state value model...')
    policy_model = policy_model.cpu()
    value_model = value_model.train()
    value_model = value_model.cuda()
    torch.cuda.empty_cache()

    value_stats = []
    for _ in range(cfg.update_epochs):
        # Split transitions into micro batches
        batch_indices = split_indices_into_bins(cfg.micro_batch_size, len(transitions), shuffle=True, drop_last=True)
        micro_batches = [get_batched_transitions([transitions[i] for i in indices], eos_id) for indices in batch_indices]

        for i in range(0, len(micro_batches), cfg.gradient_accum_steps):
            stats = update_value_model(
                micro_batches[i : i + cfg.gradient_accum_steps], value_model, value_optimizer, value_scheduler
            )
            value_stats.append(stats)

    value_model = value_model.cpu()

    return policy_stats, value_stats


def update_policy_model(
    micro_batches: List[Mapping[Text, torch.Tensor]],
    policy_model: Transformer,
    policy_optimizer: torch.optim.AdamW,
    policy_scheduler: LinearWarmupLRScheduler,
    ptx_loader: DataLoader,
) -> Mapping[Text, Any]:
    torch.cuda.empty_cache()

    policy_optimizer.zero_grad()

    stats = {
        'pg_loss': 0,
        'entropy_loss': 0,
        'ptx_loss': 0,
        'total_loss': 0,
        'entropy': 0,
        'approx_kl': 0,
    }

    def compute_pretrain_loss():
        x, y = next(iter(ptx_loader))
        x = x.cuda()
        y = y.cuda()

        y_pred = policy_model(x)

        ptx_loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), reduction='mean')
        return ptx_loss

    t0 = time.time()
    # accumulate gradients over N micro batches
    for micro_batch in micro_batches:
        tokens = (micro_batch['tokens']).to(dtype=torch.long, device='cuda')
        actions = (micro_batch['actions']).to(dtype=torch.long, device='cuda')  # Actions are discrete
        behavior_logprobs = (micro_batch['logprobs']).to(dtype=torch.float, device='cuda')
        advantages = (micro_batch['advantages']).to(dtype=torch.float, device='cuda')
        mask = (micro_batch['mask']).to(dtype=torch.float, device='cuda')

        # Given past states, get predicted action probabilities and state value
        pi_logits = policy_model(tokens)

        pi_dist = Categorical(logits=pi_logits)
        pi_logprobs = pi_dist.log_prob(actions)
        entropy = pi_dist.entropy()

        assert pi_logprobs.shape == behavior_logprobs.shape

        # Compute PPO clipped surrogate objective
        ratio = torch.exp(pi_logprobs - behavior_logprobs)
        clipped_ratio = torch.clamp(ratio, min=1.0 - cfg.policy_clip_eps, max=1.0 + cfg.policy_clip_eps)
        pg_loss = torch.min(ratio * advantages.detach(), clipped_ratio * advantages.detach())

        # apply loss mask
        assert entropy.shape == pg_loss.shape == mask.shape
        entropy = masked_mean(entropy, mask)
        pg_loss = masked_mean(pg_loss, mask)

        # Averaging over batch dimension
        # Negative sign to indicate we want to maximize the policy gradient objective function
        pg_loss = -torch.mean(pg_loss)
        entropy = torch.mean(entropy)
        entropy_loss = -cfg.entropy_coef * entropy

        loss = pg_loss + entropy_loss
        scaled_loss = loss / len(micro_batches)
        scaled_loss.backward()

        # Compute ptx loss, mixing pretraining gradients into PPO,
        # we do it separately because a single backward() call will cause CUDA OOM
        if cfg.ptx_coef > 0:
            ptx_loss = compute_pretrain_loss()
            ptx_loss = cfg.ptx_coef * torch.mean(ptx_loss)
            scaled_ptx_loss = ptx_loss / len(micro_batches)
            scaled_ptx_loss.backward()

        # track training statistics
        approx_kl = 0.5 * torch.mean(torch.square(pi_logprobs.detach() - behavior_logprobs.detach()))

        stats['pg_loss'] += pg_loss.detach().item()
        stats['entropy_loss'] += entropy_loss.detach().item()
        stats['ptx_loss'] += ptx_loss.detach().item() if cfg.ptx_coef > 0 else 0.0
        stats['total_loss'] += (loss + ptx_loss).detach().item() if cfg.ptx_coef > 0 else loss.detach().item()

        stats['entropy'] += entropy.detach().item()
        stats['approx_kl'] += approx_kl.detach().item()

    grad_norm = get_grad_norm_local(policy_model)

    if cfg.policy_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(),
            max_norm=cfg.policy_grad_clip,
            error_if_nonfinite=True,
        )

    policy_optimizer.step()
    policy_scheduler.step()

    t1 = time.time()

    stats = {k: v / len(micro_batches) for k, v in stats.items()}

    stats['learning_rate'] = policy_optimizer.param_groups[0]['lr']
    stats['grad_norm'] = grad_norm.item()
    stats['step_time'] = t1 - t0

    return stats


def update_value_model(
    micro_batches: List[Mapping[Text, torch.Tensor]],
    value_model: RewardModel,
    value_optimizer: torch.optim.AdamW,
    value_scheduler: LinearWarmupLRScheduler,
) -> Mapping[Text, Any]:
    torch.cuda.empty_cache()

    value_optimizer.zero_grad()

    stats = {
        'error': 0,
        'loss': 0,
    }

    t0 = time.time()

    # accumulate gradients over N micro batches
    for batch in micro_batches:
        tokens = (batch['tokens']).to(dtype=torch.long, device='cuda')
        values = (batch['values']).to(dtype=torch.float, device='cuda')
        returns = (batch['returns']).to(dtype=torch.float, device='cuda')
        mask = (batch['mask']).to(dtype=torch.float, device='cuda')

        # Given past states, get predicted action probabilities and state value
        pred_values = value_model(tokens).squeeze(-1)

        assert pred_values.shape == returns.shape

        # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L343C38-L347
        if cfg.value_clip_eps > 0:
            pred_values_clipped = torch.clamp(pred_values, values - cfg.value_clip_eps, values + cfg.value_clip_eps)
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
        scaled_loss = loss / len(micro_batches)
        scaled_loss.backward()

        stats['error'] += pred_error.detach().mean().item()
        stats['loss'] += loss.detach().item()

    grad_norm = get_grad_norm_local(value_model)

    if cfg.value_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            value_model.parameters(),
            max_norm=cfg.value_grad_clip,
            error_if_nonfinite=True,
        )

    value_optimizer.step()
    value_scheduler.step()

    t1 = time.time()

    stats = {k: v / len(micro_batches) for k, v in stats.items()}

    stats['learning_rate'] = value_optimizer.param_groups[0]['lr']
    stats['grad_norm'] = grad_norm.item()
    stats['step_time'] = t1 - t0

    return stats


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


# -------------------------------- RL PPO training --------------------------------


def build_policy_model(vocab_size, ckpt_file, disable_grad, strict=True) -> Transformer:
    assert vocab_size > 0

    model_args = ModelArgs.from_model_type(cfg.model_type)
    model_args.vocab_size = vocab_size
    model_args.max_seq_len = cfg.max_seq_len
    model_args.max_batch_size = cfg.selfplay_batch_size
    model_args.use_cache = False
    model_args.embed_dropout = cfg.embed_dropout
    model_args.attn_dropout = cfg.attn_dropout
    model_args.resid_dropout = cfg.resid_dropout
    model_args.head_type = 'lm_head'

    model = Transformer(model_args)

    if os.path.exists(ckpt_file):
        print(f'Loading model checkpoint {ckpt_file}...')
        model_state = torch.load(ckpt_file)
        model.load_state_dict(model_state, strict=strict)
        del model_state  # free up CPU RAM

    if disable_grad:
        for p in model.parameters():
            p.requires_grad = False

    return model


def build_reward_model(vocab_size, ckpt_file, disable_grad, strict=True) -> RewardModel:
    assert vocab_size > 0

    model_args = ModelArgs.from_model_type(cfg.model_type)
    model_args.vocab_size = vocab_size
    model_args.max_seq_len = cfg.max_seq_len
    model_args.max_batch_size = cfg.selfplay_batch_size
    model_args.use_cache = False
    model_args.embed_dropout = cfg.embed_dropout
    model_args.attn_dropout = cfg.attn_dropout
    model_args.resid_dropout = cfg.resid_dropout
    model_args.head_type = 'scalar_head'

    model = RewardModel(model_args)

    if os.path.exists(ckpt_file):
        print(f'Loading model checkpoint {ckpt_file}...')
        model_state = torch.load(ckpt_file)
        model.load_state_dict(model_state, strict=strict)
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
    assert cfg.train_episodes_per_epoch >= (cfg.gradient_accum_steps * cfg.micro_batch_size)

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

    train_ptx_dataset = BlendedDataset(
        data_sources=cfg.train_ptx_datasources,
        max_seq_len=cfg.max_seq_len,
        rank=rank,
        world_size=world_size,  # shard the dataset
        seed=int(cfg.seed + rank),
    )

    train_ptx_loader = DataLoader(train_ptx_dataset, **cuda_kwargs)

    logger.info(f'Train PTX dataset metadata:\n{train_ptx_dataset.get_metadata()}')

    train_prompt_dataset = PromptOnlyDataset(
        data_sources=cfg.train_prompt_datasources,
        max_seq_len=cfg.max_prompt_len,
        max_samples=cfg.max_train_samples,
        seed=cfg.seed,
    )

    logger.info(f'Train prompt dataset metadata:\n{train_prompt_dataset.get_metadata()}')

    val_prompt_dataset = PromptOnlyDataset(
        data_sources=cfg.val_prompt_datasources,
        max_seq_len=cfg.max_prompt_len,
        max_samples=cfg.max_val_samples,
        seed=cfg.seed,
    )

    logger.info(f'Validation prompt dataset metadata:\n{val_prompt_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    logger.info('Initialize model and optimizer...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    logger.info('Initialize SFT model...')
    sft_model = build_policy_model(vocab_size=vocab_size, ckpt_file=cfg.sft_ckpt_file, disable_grad=True)
    sft_model = sft_model.to(torch.bfloat16)
    sft_model = sft_model.eval()

    logger.info('Initialize RM model...')
    reward_model = build_reward_model(vocab_size=vocab_size, ckpt_file=cfg.rm_ckpt_file, disable_grad=True)
    reward_model = reward_model.to(torch.bfloat16)
    reward_model = reward_model.eval()

    logger.info('Initialize policy and value models...')

    with lora(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, enabled=True):
        policy_model = build_policy_model(
            vocab_size=vocab_size,
            ckpt_file=cfg.policy_ckpt_file,
            disable_grad=False,
            strict=False,
        )

        value_model = build_reward_model(
            vocab_size=vocab_size,
            ckpt_file=cfg.value_ckpt_file,
            disable_grad=False,
            strict=False,
        )

    mark_only_lora_as_trainable(policy_model, train_bias=cfg.train_bias, train_head=cfg.train_head)
    mark_only_lora_as_trainable(value_model, train_bias=cfg.train_bias, train_head=cfg.train_head)
    trainable_model_to_dtype(policy_model)
    trainable_model_to_dtype(value_model)

    logger.info('Initialize optimizer...')

    policy_optimizer = create_optimizer(
        model=policy_model,
        lr=cfg.policy_init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        use_bnb_8bit=cfg.use_bnb_8bit,
    )

    policy_scheduler = LinearWarmupLRScheduler(
        optimizer=policy_optimizer, init_lr=cfg.policy_init_lr, max_lr=cfg.policy_max_lr, warmup_steps=cfg.policy_warmup_steps
    )
    value_optimizer = create_optimizer(
        model=value_model,
        lr=cfg.value_init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        use_bnb_8bit=cfg.use_bnb_8bit,
    )
    value_scheduler = LinearWarmupLRScheduler(
        optimizer=value_optimizer, init_lr=cfg.value_init_lr, max_lr=cfg.value_max_lr, warmup_steps=cfg.value_warmup_steps
    )

    reward_stats = RunningMeanStd()

    # --------------- Start Training ---------------

    if cfg.warmup_episodes > 0:
        logger.info(f'Start to run {cfg.warmup_episodes} warm up episodes...')
        run_warmup_episodes(
            prompt_dataset=train_prompt_dataset,
            tokenizer=tokenizer,
            policy_model=policy_model,
            reward_model=reward_model,
            reward_stats=reward_stats,
            num_episodes=cfg.warmup_episodes,
            batch_size=cfg.selfplay_batch_size,
            temperature=cfg.train_temperature,
            top_p=cfg.train_top_p,
        )

    tb_writer = None
    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

    num_epochs = cfg.max_episodes // cfg.train_episodes_per_epoch

    episode_count = 0
    train_step_count = 0
    for epoch in range(1, num_epochs + 1):
        logger.info(f'Epoch {epoch}')
        logger.info(f'Start to generate {cfg.train_episodes_per_epoch} train episodes...')
        transitions, train_episode_stats, train_epoch_stats = run_selfplay_episodes(
            prompt_dataset=train_prompt_dataset,
            tokenizer=tokenizer,
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            sft_model=sft_model,
            num_episodes=cfg.train_episodes_per_epoch,
            batch_size=cfg.selfplay_batch_size,
            temperature=cfg.train_temperature,
            top_p=cfg.train_top_p,
            reward_stats=reward_stats,
            update_reward_stats=True,
        )

        # logging - selfplay epoch statistics
        if tb_writer is not None:
            for k, v in train_epoch_stats.items():
                if isinstance(v, (int, float)):
                    tb_writer.add_scalar(f'train_epochs/{k}', v, epoch)
        logger.info(f'Train epoch stats: {train_epoch_stats}')

        # logging - selfplay episode statistics
        for stats in train_episode_stats:
            episode_count += 1
            if episode_count == 1 or episode_count % cfg.selfplay_log_interval == 0:
                if tb_writer is not None:
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(f'train_episodes/{k}', v, episode_count)

        logger.info(f'Start to train the agent using {len(transitions)} selfplay episodes...')
        policy_stats, value_stats = run_ppo_training_steps(
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            policy_scheduler=policy_scheduler,
            value_model=value_model,
            value_optimizer=value_optimizer,
            value_scheduler=value_scheduler,
            transitions=transitions,
            ptx_loader=train_ptx_loader,
            eos_id=tokenizer.eos_id,
        )

        assert len(policy_stats) == len(value_stats)

        # logging - training step statistics
        for pi_stats, v_stats in zip(policy_stats, value_stats):
            train_step_count += 1
            if train_step_count == 1 or train_step_count % cfg.train_log_interval == 0:
                if tb_writer is not None:
                    for k, v in pi_stats.items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(f'train_policy/{k}', v, train_step_count)
                    for k, v in v_stats.items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(f'train_value/{k}', v, train_step_count)

        # validation episodes
        if epoch % cfg.var_interval == 0:
            logger.info(f'Start to generate {cfg.val_episodes_per_epoch} validation episodes...')
            _, val_episode_stats, val_epoch_stats = run_selfplay_episodes(
                prompt_dataset=val_prompt_dataset,
                tokenizer=tokenizer,
                policy_model=policy_model,
                value_model=None,
                reward_model=reward_model,
                sft_model=sft_model,
                num_episodes=cfg.val_episodes_per_epoch,
                batch_size=cfg.selfplay_batch_size,
                temperature=cfg.val_temperature,
                top_p=cfg.val_top_p,
                reward_stats=reward_stats,
            )

            # logging - validation epoch statistics
            if tb_writer is not None:
                for k, v in val_epoch_stats.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f'val_epochs/{k}', v, epoch)
            logger.info(f'Validation epoch stats: {val_epoch_stats}')

            # checkpointing
            policy_checkpoint = lora_state_dict(policy_model, train_bias=cfg.train_bias, train_head=cfg.train_head)
            torch.save(policy_checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}_policy-epoch-{epoch}.pth'))
            value_checkpoint = lora_state_dict(value_model, train_bias=cfg.train_bias, train_head=cfg.train_head)
            torch.save(value_checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}_value-epoch-{epoch}.pth'))

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
