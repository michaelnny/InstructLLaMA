# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Run pre-training without using any previous model weights."""
import os
import math
from typing import Tuple, Mapping, Text, Any
import tqdm
import random
from functools import partial
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    StateDictType,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model import Transformer, TransformerBlock, ModelArgs
from instruct_llama.models.tokenizer import Tokenizer

from instruct_llama.configs.pretrain import config as cfg
from instruct_llama.core.custom_dataset import BlendedDataset

from instruct_llama.core.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.core.train_helper import (
    create_trace_profiler,
    create_optimizer,
    compute_num_trainable_params,
    get_grad_norm_fsdp,
)
from instruct_llama.utils.logger import create_logger, log_statistics
from instruct_llama.utils.tracker import StatsTracker
from instruct_llama.utils.checkpoint import create_fsdp_full_checkpoint


# ---------------------------------------- FSDP module ----------------------------------------
local_rank = int(os.environ['LOCAL_RANK'])
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])


def setup():
    # initialize the process group
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    torch.cuda.empty_cache()


# FSDP activation checkpointing
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_activation_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    check_fn = lambda submodule: isinstance(submodule, TransformerBlock)  # noqa: E731

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


# FSDP auto wrap
fsdp_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})


def get_fsdp_mixed_precision_policy(compute_dtype) -> Tuple[ShardedGradScaler, MixedPrecision]:
    scaler = None
    mixed_precision_policy = None  # defaults to fp32

    if compute_dtype == torch.bfloat16:
        print('bFloat16 enabled for FSDP mixed precision ...')
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )
    elif compute_dtype == torch.float16:
        print('float16 enabled for FSDP mixed precision ...')
        # requires grad scaler
        scaler = ShardedGradScaler()
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            # Gradient communication precision.
            reduce_dtype=torch.float16,
            # Buffer precision.
            buffer_dtype=torch.float16,
        )

    return scaler, mixed_precision_policy


# ---------------------------------------- FSDP module ----------------------------------------


def compute_pretrain_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0]

    B, T, *_ = logits.shape
    losses = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
    losses = losses.view(B, T)  # [batch_size, seq_len]
    losses = losses.mean(1)  # [batch_size]
    return losses


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, float]:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0]

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)

    num_accurate = pred.eq(targets.view_as(pred)).sum().item()
    num_samples = targets.shape[0] * targets.shape[1]

    return num_accurate, num_samples


def train_step(
    model: Transformer,
    batch: Tuple[torch.Tensor],
    scaler: ShardedGradScaler,
    gradient_accum_steps: int,
    local_rank: int,
    tracker: StatsTracker,
) -> None:
    """Run a single training step, where we do a forward + backward passes, but do no update parameters"""

    assert gradient_accum_steps >= 1

    x, y = batch
    x, y = (
        x.to(local_rank, non_blocking=True),
        y.to(local_rank, non_blocking=True),
    )

    output = model(x)
    losses = compute_pretrain_loss(output, y)
    loss = losses.mean()
    # scale the loss to account for gradient accumulation
    scaled_loss = loss / gradient_accum_steps

    if scaler is not None:  # when using float16
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    num_acc, num_samples = compute_metrics(output.detach(), y.detach())
    tracker.update(losses.detach(), num_acc, num_samples)


def update_step(
    model: Transformer,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    grad_clip: float,
    scaler: ShardedGradScaler = None,
) -> None:
    """Run a single parameter update step"""
    if grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        # FSDP needs to use this method to clip gradient norm instead of torch.nn.utils.clip_grad_norm_
        model.clip_grad_norm_(grad_clip)

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
    rank: int,
    local_rank: int,
    tracker: StatsTracker,
) -> Mapping[Text, Any]:
    """Run M validation steps"""

    tracker.reset()
    val_pbar = None
    if rank == 0:
        val_pbar = tqdm.tqdm(range(steps), colour='green', desc='Validation steps')

    for i, (x, y) in enumerate(loader):
        x, y = (
            x.to(local_rank, non_blocking=True),
            y.to(local_rank, non_blocking=True),
        )

        output = model(x)
        losses = compute_pretrain_loss(output, y)
        num_acc, num_samples = compute_metrics(output.detach(), y.detach())
        tracker.update(losses.detach(), num_acc, num_samples)

        if val_pbar is not None:
            val_pbar.update(1)

        if i + 1 >= steps:
            break

    if val_pbar is not None:
        val_pbar.close()


def init_weights(model: Transformer) -> None:
    for m_name, module in model.named_modules():
        if m_name.endswith('wo.weight') and isinstance(module, torch.nn.Linear):
            # apply special scaled init to the residual projections
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * model.n_layers))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def fsdp_main():
    assert cfg.train_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 0
    assert cfg.val_steps >= 1

    if cfg.checkpoint_type != StateDictType.FULL_STATE_DICT:
        raise ValueError('This script only supports FSDP FULL_STATE_DICT checkpoint.')

    if not torch.version.cuda:
        raise RuntimeError('This script requires Pytorch with CUDA.')

    setup()

    # --------------- Load datasets ---------------

    logger = create_logger(rank=rank)

    logger.info('Loading datasets ...')

    tokenizer = Tokenizer(cfg.tokenizer_file)

    train_dataset = BlendedDataset(
        data_sources=cfg.train_datasources,
        max_seq_len=cfg.max_seq_len,
        rank=rank,
        world_size=world_size,  # shard the dataset
        seed=cfg.seed,
    )

    # Our custom IterableDatasets already have sharding and shuffle mechanism implemented
    cuda_kwargs = {
        'num_workers': cfg.dataloader_workers,
        'pin_memory': False,
        'shuffle': False,
        'sampler': None,
    }
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, **cuda_kwargs)
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = BlendedDataset(
            data_sources=cfg.val_datasources,
            max_seq_len=cfg.max_seq_len,
            rank=rank,
            world_size=world_size,  # shard the dataset
            seed=cfg.seed,
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, **cuda_kwargs)
        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    batch_size = int(cfg.train_batch_size * cfg.gradient_accum_steps)
    steps_per_epoch = len(train_loader) // cfg.gradient_accum_steps
    max_train_steps = steps_per_epoch * cfg.num_epochs

    # --------------- Setup model and optimizer ---------------

    logger.info('Initializing model and optimizer ...')

    torch.cuda.set_device('cuda:0')
    clear_gpu_cache()

    compute_dtype = torch.float32
    if cfg.mixed_precision:
        if torch.version.cuda and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
    else:
        logger.warning('Training in float32 mode, make sure you have enough GPU RAM')

    model_args = ModelArgs.from_model_type(
        model_type=cfg.model_type,
        head_type=cfg.head_type,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.max_seq_len,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
        head_dropout=cfg.head_dropout,
    )

    assert model_args.head_type == 'lm_head'

    model = Transformer(model_args)

    init_weights(model)

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    model.to(compute_dtype)

    # mix precision policy
    scaler, mixed_precision_policy = get_fsdp_mixed_precision_policy(compute_dtype)

    model = FSDP(
        model,
        auto_wrap_policy=fsdp_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=cfg.backward_prefetch,
        forward_prefetch=cfg.forward_prefetch,
        cpu_offload=CPUOffload(offload_params=cfg.cpu_offload),
        sharding_strategy=cfg.sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=cfg.limit_all_gathers,
        use_orig_params=cfg.use_orig_params,
    )

    if cfg.fsdp_activation_checkpointing:
        logger.info('applying FSDP activation checkpointing ...')
        apply_fsdp_activation_checkpointing(model)

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
        paged_adamw=False,
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

    torch_profiler = None
    tb_writer = None
    train_pbar = None
    best_val_accuracy = 0.0
    train_steps = 0

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        train_pbar = tqdm.tqdm(range(max_train_steps), colour='blue', desc='Training steps')
        tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

        # Careful as the logs will grow very fast
        if cfg.use_profiler:
            torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    train_tracker = StatsTracker(local_rank=local_rank, world_size=world_size)
    val_tracker = StatsTracker(local_rank=local_rank, world_size=world_size)

    logger.info(f'Starting to run {cfg.num_epochs} training epochs, total of {max_train_steps} steps, with batch size {batch_size}')

    for epoch in range(1, cfg.num_epochs + 1):  # for each epoch
        logger.info(f'Start epoch {epoch}')
        model.train()
        train_tracker.reset()
        val_tracker.reset()

        for iter, batch in enumerate(train_loader):  # for each batch in current epoch
            train_step(model, batch, scaler, cfg.gradient_accum_steps, local_rank, train_tracker)

            if iter % cfg.gradient_accum_steps == 0:
                grad_norm = get_grad_norm_fsdp(model, rank, world_size, cfg.sharding_strategy)
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
                    create_fsdp_full_checkpoint(model, rank, os.path.join(cfg.ckpt_dir, f'{cfg.model_type}-steps-{train_steps}.pth'))

                # validation steps
                if cfg.val_steps > 0 and (cfg.val_interval > 0 and train_steps % cfg.val_interval == 0 or train_steps == max_train_steps):
                    model.eval()
                    run_validation_steps(model, val_loader, cfg.val_steps, rank, local_rank, val_tracker)
                    model.train()

                    val_stats = val_tracker.get_dict(reset=True)
                    log_statistics(tb_writer, train_steps, val_stats, False)

                    # save best model
                    if val_stats['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_stats['accuracy']
                        logger.info(f'New best validation accuracy: {val_stats["accuracy"]:.4f}')
                        create_fsdp_full_checkpoint(model, rank, os.path.join(cfg.ckpt_dir, f'{cfg.model_type}-best.pth'))

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    fsdp_main()
