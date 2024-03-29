# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Train reward model (RM) full-scale, starting from a fine-tuned model.
with the option to frozen first N decoder layers in the model.
"""
import os
import functools
from typing import List, Tuple, Mapping, Text, Any
import tqdm
import random
import math
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model import Transformer, ModelArgs
from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.configs.rm import config as cfg
from instruct_llama.core.custom_dataset import ComparisonsDataset
from instruct_llama.core.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.core.train_helper import (
    make_model_layer_trainable,
    create_trace_profiler,
    create_optimizer,
    compute_num_trainable_params,
    get_grad_norm_local,
    optimizer_to,
)
from instruct_llama.utils.logger import create_logger, log_statistics
from instruct_llama.utils.tracker import RMStatsTracker
from instruct_llama.utils.normalizer import Normalizer
from instruct_llama.utils.checkpoint import create_checkpoint, create_normalizer_checkpoint


def clear_gpu_cache(rank=None):
    torch.cuda.empty_cache()


def extract_rewards_from_groups(rewards: torch.Tensor, group_indices: List[List[int]]) -> List[torch.Tensor]:
    """Get chosen and rejected rewards for a list of sample based on the group indices"""
    results = []

    for indices in group_indices:
        results.append(rewards[indices, ...])

    return results


def compute_rm_comparison_loss(rewards: torch.Tensor) -> torch.Tensor:
    """Compute RM comparison loss for N responses.

    Note we assume the rewards are for the ordered completions for a given prompt,
    where the best completion is the first, and worst completion is the last.
    """
    assert len(rewards.shape) == 1  # [num_completions]

    losses = []
    N = len(rewards)  # number of completions
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2
    assert C >= 1

    # for each 'better' completion 0, 1, ..., compare to the remaining of rejected completions
    # for example:
    # 0 <-> (1, 2, ..., N-1)
    # 1 <-> (2, 3, ..., N-1)

    for i in range(0, N - 1):
        r_rejected = rewards[i + 1 :]
        r_preferred = rewards[i].repeat(len(r_rejected))
        assert r_preferred.shape == r_rejected.shape
        loss = -torch.nn.functional.logsigmoid(r_preferred - r_rejected).view(-1)
        losses.append(loss)

    losses = torch.cat(losses, dim=0).view(-1)
    assert len(losses) == C
    # average over number of combinations
    losses = losses.sum() / C
    return losses


def compute_batch_rm_comparison_loss(rewards: torch.Tensor, group_indices: List[List[int]]) -> torch.Tensor:
    # Group chosen and reject rewards from the same sample
    reward_groups = extract_rewards_from_groups(rewards, group_indices)

    losses = []
    for item in reward_groups:
        losses.append(compute_rm_comparison_loss(item))

    return torch.stack(losses, dim=0)


@torch.no_grad()
def compute_metrics(rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute number of accurate predictions in terms of reward values.

    Note we assume the rewards are for the ordered completions for a given prompt,
    where the best completion is the first, and worst completion is the last.
    """
    assert len(rewards.shape) == 1

    N = len(rewards)  # number of responses
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2

    chosen_rewards = []
    rejected_rewards = []

    # for each 'better' completion, compare to the remaining of rejected completions
    for i in range(0, N - 1):
        r_rejected = rewards[i + 1 :]
        rejected_rewards.append(r_rejected)
        chosen_rewards.append(rewards[i].repeat(len(r_rejected)))

    chosen_rewards = torch.cat(chosen_rewards, dim=0).view(-1)
    rejected_rewards = torch.cat(rejected_rewards, dim=0).view(-1)
    assert len(chosen_rewards) == len(rejected_rewards) == C
    return chosen_rewards, rejected_rewards


@torch.no_grad()
def compute_batch_metrics(rewards: torch.Tensor, group_indices: List[List[int]]) -> torch.Tensor:
    # Group chosen and reject rewards from the same sample
    reward_groups = extract_rewards_from_groups(rewards, group_indices)

    chosen_rewards, rejected_rewards = [], []
    for item in reward_groups:
        r_chosen, r_rejected = compute_metrics(item)
        chosen_rewards.append(r_chosen)
        rejected_rewards.append(r_rejected)

    return torch.cat(chosen_rewards, dim=0), torch.cat(rejected_rewards, dim=0)


def train_step(
    model: Transformer,
    batch: Tuple[torch.Tensor],
    scaler: torch.cuda.amp.GradScaler,
    gradient_accum_steps: int,
    tracker: RMStatsTracker,
) -> None:
    """Run a single training step, where we do a forward + backward passes, but do no update parameters"""
    assert gradient_accum_steps >= 1

    batch_tokens, terminal_steps, group_indices = batch
    batch_tokens = batch_tokens.to('cuda', non_blocking=True)
    terminal_steps = terminal_steps.to('cuda', non_blocking=True)

    # forward pass to compute reward for all completions
    outputs = model(batch_tokens).squeeze(-1)  # [num_combinations, seq_length]

    # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
    rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [num_combinations]

    # compute loss in a single go
    losses = compute_batch_rm_comparison_loss(rewards, group_indices)
    loss = losses.mean()
    # scale the loss to account for gradient accumulation
    scaled_loss = loss / gradient_accum_steps

    if scaler is not None:  # when using float16
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    chosen_rewards, rejected_rewards = compute_batch_metrics(rewards.detach(), group_indices)
    tracker.update(losses.detach(), chosen_rewards, rejected_rewards)


def update_step(
    model: Transformer,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler = None,
) -> None:
    """Run a single parameter update step"""
    if grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if scaler is not None:  # when using float16
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        optimizer.step()

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch


@torch.no_grad()
def run_validation_steps(
    model: Transformer,
    loader: DataLoader,
    steps: int,
    tracker: RMStatsTracker,
    normalizer: Normalizer = None,
    save_normalizer_dir: str = None,
    save_normalizer_interval: int = 0,
) -> None:
    """Run M validation steps"""

    if tracker is not None:
        tracker.reset()

    val_pbar = tqdm.tqdm(range(steps), colour='green', desc='Validation steps')

    should_save_normalizer = True if normalizer is not None and save_normalizer_dir is not None and save_normalizer_interval > 0 else False

    # one batch contains prompt and a list of completions for possible multiple samples,
    # where the completions are already ordered by the score, from best to worst
    for i, (batch_tokens, terminal_steps, group_indices) in enumerate(loader):
        batch_tokens = batch_tokens.to('cuda', non_blocking=True)
        terminal_steps = terminal_steps.to('cuda', non_blocking=True)

        # forward pass to compute reward for all completions
        outputs = model(batch_tokens).squeeze(-1)  # [num_combinations, seq_length]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [num_combinations]

        if normalizer is not None:
            normalizer.update(rewards)

        # compute loss in a single go
        if tracker is not None:
            losses = compute_batch_rm_comparison_loss(rewards, group_indices)
            chosen_rewards, rejected_rewards = compute_batch_metrics(rewards.detach(), group_indices)
            tracker.update(losses.detach(), chosen_rewards, rejected_rewards)

        val_pbar.update(1)

        if i > 0 and should_save_normalizer and (i % save_normalizer_interval == 0 or i + 1 >= steps):
            total_samples = (i + 1) * len(rewards)
            create_normalizer_checkpoint(normalizer, os.path.join(save_normalizer_dir, f'normalizer-samples-{total_samples}.pth'))

        if i + 1 >= steps:
            break

    val_pbar.close()


def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens_list = []
    for item in batch:  # for each sample in current batch
        # Note we assume the tokens are for the ordered completions for a given prompt,
        # where the best completion is the first, and worst completion is the last.
        for d in item:  # for each response in current sample
            tokens_list.append(d)

    batch_size = len(tokens_list)

    # each item might have different number of completion pairs,
    # we store a group indices so later can use it to separate them when computing loss
    groups_indices = []

    indices = list(range(batch_size))
    for i in range(len(batch)):
        num_completions = len(batch[i])
        group = indices[:num_completions]
        indices = indices[num_completions:]
        groups_indices.append(group)

    assert len(groups_indices) == len(batch)

    max_batch_seq_len = max([len(d) for d in tokens_list])
    assert max_batch_seq_len <= max_seq_len

    if full_pad:
        max_batch_seq_len = max_seq_len

    # concatenate prompt, completion from multiple samples together
    batch_sequences = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)

    # record the terminal index of the completion, often referred to as the terminal time step in RL
    terminal_steps = torch.zeros((batch_size), dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        seq = torch.tensor(tokens, dtype=torch.long)
        seq_len = len(seq)

        batch_sequences[i, :seq_len] = seq
        # -1 because indexing starts from zero
        terminal_steps[i] = seq_len - 1

    return batch_sequences, terminal_steps, groups_indices


def init_head_weights(model: Transformer) -> None:

    if hasattr(model, 'scalar_head'):
        module = model.scalar_head

        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * model.n_layers))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def main():
    assert cfg.num_epochs >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 0
    assert cfg.val_steps >= 1
    assert cfg.frozen_layers >= 0

    if not torch.version.cuda:
        raise RuntimeError('This script requires Pytorch with CUDA.')

    if not os.path.exists(cfg.reward_ckpt_file):
        raise ValueError(f'Invalid model checkpoint "{cfg.reward_ckpt_file}", aborting ...')

    # --------------- Load datasets ---------------
    logger = create_logger()

    logger.info('Loading datasets ...')

    tokenizer = Tokenizer(cfg.tokenizer_file)

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=cfg.max_seq_len,
        full_pad=cfg.full_pad,
    )

    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': cfg.dataloader_workers,
        'pin_memory': False,
        'shuffle': True,
    }

    train_dataset = ComparisonsDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)
    train_kwargs = {'batch_size': cfg.train_batch_size, 'sampler': None}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(dataset=train_dataset, **train_kwargs)
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = ComparisonsDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)
        val_kwargs = {'batch_size': cfg.val_batch_size, 'sampler': None}
        val_kwargs.update(cuda_kwargs)
        val_loader = DataLoader(dataset=val_dataset, **val_kwargs)
        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    batch_size = int(cfg.train_batch_size * cfg.gradient_accum_steps)
    steps_per_epoch = len(train_loader) // cfg.gradient_accum_steps
    max_train_steps = steps_per_epoch * cfg.num_epochs

    # --------------- Setup model and optimizer ---------------

    logger.info('Initializing model and optimizer ...')

    torch.cuda.set_device('cuda:0')
    clear_gpu_cache()

    compute_dtype = torch.float32
    scaler = None
    if cfg.mixed_precision:
        if torch.version.cuda and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    else:
        logger.info('Training in float32 mode, make sure you have enough GPU RAM')

    model_args = ModelArgs.from_model_type(
        model_type=cfg.model_type,
        # Regular configurations
        head_type='scalar_head',
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.max_seq_len,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
        head_dropout=cfg.head_dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    model = Transformer(model_args)

    # Load model checkpoint using strict=False,
    # because there's not scalar head weights in the checkpoint state
    if os.path.exists(cfg.reward_ckpt_file):
        logger.info(f'Loading RM checkpoint {cfg.reward_ckpt_file} ...')
        model_state = torch.load(cfg.reward_ckpt_file)
        model.load_state_dict(model_state, strict=False)
        del model_state

    if cfg.random_head_weights:
        logger.info('Initializing weights for scalar head ...')
        init_head_weights(model)

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        module = module.to(dtype=compute_dtype)

    # freeze first N decoder layers and make last M-N decoder layers and output layer trainable
    trainable_layers = ['post_norm', 'scalar_head']
    for i in range(cfg.frozen_layers, model.n_layers):
        if cfg.train_atten_qv_layers_only:
            # only train the attention query and value layers in the encoder block, this can save more GPU resource
            trainable_layers.append(f'layers.{i}.attention.wq')
            trainable_layers.append(f'layers.{i}.attention.wv')
        else:
            # train the entire encoder block
            trainable_layers.append(f'layers.{i}')

    # logger.info(f'Trainable layers:\n{trainable_layers}')
    make_model_layer_trainable(model, trainable_layers)

    model = model.to('cuda')

    torch.cuda.empty_cache()

    logger.info('Initializing optimizer ...')

    num_trainable, num_frozen = compute_num_trainable_params(model)
    logger.info(f'Number of trainable parameters: {num_trainable:,}')
    logger.info(f'Number of frozen parameters: {num_frozen:,}')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        paged_adamw=cfg.use_paged_adamw,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=cfg.init_lr,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=int(cfg.warmup_ratio * max_train_steps),
        max_decay_steps=max_train_steps,
    )

    # --------------- Start Training ---------------

    log_dir = os.path.join(cfg.log_dir, cfg.model_type)
    ckpt_dir = os.path.join(cfg.ckpt_dir, cfg.model_type)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    torch_profiler = None
    tb_writer = SummaryWriter(log_dir)
    train_pbar = tqdm.tqdm(range(max_train_steps), colour='blue', desc='Training steps')
    best_val_loss = np.inf
    train_steps = 0

    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, f'{cfg.model_type}_profile_traces'))

    train_tracker = RMStatsTracker()
    val_tracker = RMStatsTracker()

    logger.info(f'Starting to run {cfg.num_epochs} training epochs, total of {max_train_steps} steps, with batch size {batch_size}')

    for epoch in range(1, cfg.num_epochs + 1):  # for each epoch
        logger.info(f'Start epoch {epoch}')
        model.train()
        train_tracker.reset()
        val_tracker.reset()

        for iter, batch in enumerate(train_loader):  # for each batch in current epoch
            train_step(model, batch, scaler, cfg.gradient_accum_steps, train_tracker)

            if iter % cfg.gradient_accum_steps == 0:
                grad_norm = get_grad_norm_local(model)
                update_step(model, optimizer, scheduler, cfg.grad_clip, scaler)
                train_pbar.update(1)
                train_steps += 1

                if torch_profiler is not None:
                    torch_profiler.step()

                train_stats = train_tracker.get_dict(reset=True)
                # logging training statistics
                if train_steps % cfg.log_interval == 0:
                    train_stats['learning_rate'] = optimizer.param_groups[0]['lr']
                    train_stats['grad_norm'] = grad_norm.item()
                    log_statistics(tb_writer, train_steps, train_stats, True)

                # regular checkpointing
                if cfg.ckpt_interval > 0 and (train_steps % cfg.ckpt_interval == 0 or train_steps == max_train_steps):
                    create_checkpoint(model=model, full_path=os.path.join(ckpt_dir, f'{cfg.model_type}-steps-{train_steps}.pth'))

                # validation steps
                if cfg.val_steps > 0 and (cfg.val_interval > 0 and train_steps % cfg.val_interval == 0 or train_steps == max_train_steps):
                    model.eval()
                    optimizer_to(optimizer, 'cpu')  # move optimizer to cpu so we can use larger batch size for validation
                    run_validation_steps(model, val_loader, cfg.val_steps, val_tracker)
                    model.train()
                    optimizer_to(optimizer, 'cuda')

                    val_stats = val_tracker.get_dict(reset=True)
                    log_statistics(tb_writer, train_steps, val_stats, False)

                    # save best model
                    if val_stats['loss'] < best_val_loss:
                        best_val_loss = val_stats['loss']
                        logger.info(f'New best validation loss: {best_val_loss:.4f}')
                        create_checkpoint(model=model, full_path=os.path.join(ckpt_dir, f'{cfg.model_type}-best.pth'))

    # create a final checkpoint
    create_checkpoint(model=model, full_path=os.path.join(ckpt_dir, f'{cfg.model_type}-steps-{train_steps}.pth'))

    # run some steps to collect statistics for reward normalizer
    norm_ckpt_dir = os.path.join(cfg.ckpt_dir, 'normalizer')
    os.makedirs(norm_ckpt_dir, exist_ok=True)

    ds_kwargs = {'batch_size': cfg.val_batch_size, 'sampler': None}
    ds_kwargs.update(cuda_kwargs)
    ds_loader = DataLoader(dataset=train_dataset, **ds_kwargs)
    reward_normalizer = Normalizer(target_mean=0.0, target_std=1.0, window_size=len(train_dataset))
    logger.info('Run steps to collect statistics for reward normalizer...')

    model.eval()
    optimizer_to(optimizer, 'cpu')
    run_validation_steps(
        model,
        ds_loader,
        len(ds_loader),
        None,
        reward_normalizer,
        norm_ckpt_dir,
        len(ds_loader) // 5,
    )


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
