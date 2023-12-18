# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Run pre-training without using any previous model weights."""
import os
import itertools
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


from instruct_llama.model import Transformer, TransformerBlock, ModelArgs
from instruct_llama.configs.pretrain import config as cfg
from instruct_llama.utils.custom_dataset import BlendedDataset
from instruct_llama.tokenizer import Tokenizer
from instruct_llama.utils.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.utils.train_helper import (
    create_trace_profiler,
    create_optimizer,
    compute_num_trainable_params,
    get_grad_norm_fsdp,
)
from instruct_llama.utils.logging import create_logger
from instruct_llama.utils.checkpoint import create_fsdp_full_checkpoint


# ---------------------------------------- FSDP module ----------------------------------------
local_rank = int(os.environ['LOCAL_RANK'])
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

logger = create_logger(rank=rank)


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


def get_fsdp_mixed_percision_policy(compute_dtype) -> Tuple[ShardedGradScaler, MixedPrecision]:
    scaler = None
    mixed_precision_policy = None  # defaults to fp32

    if compute_dtype == torch.bfloat16:
        logger.info('bFloat16 enabled for FSDP mixed precision ...')
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )
    elif compute_dtype == torch.float16:
        logger.info('float16 enabled for FSDP mixed precision ...')
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


def compute_pre_train_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0]

    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


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


def run_single_train_step(
    model: Transformer,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    gradient_accum_steps: int,
    grad_clip: float,
    return_stats: bool = False,
) -> Mapping[Text, Any]:
    """A single training iteration consists of N micro batch * M gradient accumulation steps.

    ```
    optimizer.zero_grad()
    for step in range(gradient_accum_steps):
        data, target = next(iter(train_loader))
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()

    optimizer.step()
    ```

    """
    assert gradient_accum_steps >= 1

    metrics = torch.zeros(5).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    for x, y in itertools.islice(train_loader, gradient_accum_steps):
        x, y = (
            x.to('cuda', non_blocking=True),
            y.to('cuda', non_blocking=True),
        )

        output = model(x)

        loss = compute_pre_train_loss(output, y)

        # scale the loss to account for gradient accumulation
        scaled_loss = loss / gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if return_stats:
            num_acc, num_samples = compute_metrics(output, y)
            metrics[0] += loss.item()  # sum up batch loss
            metrics[1] += np.exp(loss.item())  # sum up perplexity
            metrics[2] += 1  # increase number of micro batches
            metrics[3] += num_acc  # sum up number of accurate prediction tokens
            metrics[4] += num_samples  # sum up number of tokens

    grad_norm = get_grad_norm_fsdp(model, rank, world_size, cfg.sharding_strategy)

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

    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    if return_stats:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        train_loss = metrics[0] / metrics[2]
        train_perplexity = metrics[1] / metrics[2]
        train_accuracy = 100 * metrics[3] / metrics[4]

        return {
            'loss': train_loss.item(),
            'accuracy': train_accuracy.item(),
            'perplexity': train_perplexity.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item(),
        }
    else:
        return None


@torch.no_grad()
def run_validation_steps(
    model: Transformer,
    val_loader: DataLoader,
    val_iters: int,
) -> Mapping[Text, Any]:
    """Run M validation iterations"""
    model.eval()  # set model in validation mode

    metrics = torch.zeros(5).to(local_rank)

    inner_pbar = None
    if rank == 0:
        inner_pbar = tqdm.tqdm(range(val_iters), colour='green', desc='validation iterations')

    for x, y in itertools.islice(val_loader, val_iters):
        x, y = (
            x.to('cuda', non_blocking=True),
            y.to('cuda', non_blocking=True),
        )

        output = model(x)

        loss = compute_pre_train_loss(output, y)
        num_acc, num_samples = compute_metrics(output, y)
        metrics[0] += loss.item()  # sum up batch loss
        metrics[1] += np.exp(loss.item())  # sum up perplexity
        metrics[2] += 1  # increase number of micro batches
        metrics[3] += num_acc  # sum up number of accurate prediction tokens
        metrics[4] += num_samples  # sum up number of tokens

        if inner_pbar is not None:
            inner_pbar.update(1)

    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    val_loss = metrics[0] / metrics[2]
    val_perplexity = metrics[1] / metrics[2]
    val_accuracy = 100 * metrics[3] / metrics[4]

    inner_pbar.close()

    model.train()  # set model in training mode after validation runs

    return {'loss': val_loss.item(), 'accuracy': val_accuracy.item(), 'perplexity': val_perplexity.item()}


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
    assert cfg.num_train_iters >= 1
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 1
    assert cfg.val_iters >= 1

    batch_size = int(cfg.micro_batch_size * cfg.gradient_accum_steps)

    assert batch_size >= 1

    if cfg.checkpoint_type != StateDictType.FULL_STATE_DICT:
        raise ValueError('This script only supports FSDP FULL_STATE_DICT checkpoint.')

    if not torch.version.cuda:
        raise RuntimeError('This script requires Pytorch with CUDA.')

    setup()

    # --------------- Load datasets ---------------

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
        'pin_memory': True,
        'shuffle': False,
        'sampler': None,
    }
    train_loader = DataLoader(train_dataset, batch_size=cfg.micro_batch_size, **cuda_kwargs)
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
    )

    assert model_args.head_type == 'lm_head'

    model = Transformer(model_args)

    init_weights(model)

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    model.to(compute_dtype)

    # mix precision policy
    scaler, mixed_precision_policy = get_fsdp_mixed_percision_policy(compute_dtype)

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

    # seems not so helpful in terms of speed improvement
    if cfg.compile_model:
        logger.info('compile model using torch.compile() ...')
        model = torch.compile(model)

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
        warmup_steps=int(cfg.warmup_ratio * cfg.num_train_iters),
        max_decay_steps=cfg.num_train_iters,
    )

    # --------------- Start Training ---------------

    logger.info(f'Starting to run {cfg.num_train_iters} training iterations, with batch size {batch_size}')

    torch_profiler = None
    tb_writer = None
    inner_pbar = None
    best_val_accuracy = 0.0

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

        # Careful as the logs will grow very fast
        if cfg.use_profiler:
            torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

        inner_pbar = tqdm.tqdm(range(cfg.num_train_iters), colour='blue', desc='Training iterations')

    model.train()
    for i in range(1, cfg.num_train_iters + 1):
        train_stats = run_single_train_step(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            gradient_accum_steps=cfg.gradient_accum_steps,
            grad_clip=cfg.grad_clip,
            return_stats=i == 1 or i % cfg.log_interval == 0 or i == cfg.num_train_iters,
        )

        if inner_pbar is not None:
            inner_pbar.update(1)

        if torch_profiler is not None:
            torch_profiler.step()

        # logging
        if train_stats is not None:
            logger.info(
                f'Training iteration {i}: train loss: {train_stats["loss"]:.4f}, '
                f'train accuracy: {train_stats["accuracy"]:.2f}%, train perplexity: {train_stats["perplexity"]:.2f}, learning rate: {train_stats["learning_rate"]:.7f}'
            )

            if tb_writer is not None:
                for k, v in train_stats.items():
                    tb_writer.add_scalar(f'train/{k}', v, i)

        # regular checkpointing
        if cfg.ckpt_interval > 0 and (i % cfg.ckpt_interval == 0 or i == cfg.num_train_iters):
            create_fsdp_full_checkpoint(model, rank, os.path.join(cfg.ckpt_dir, f'{cfg.model_type}-iter-{i}.pth'))

        # validation steps
        if cfg.val_iters > 0 and (cfg.val_interval > 0 and i % cfg.val_interval == 0 or i == cfg.num_train_iters):
            val_stats = run_validation_steps(
                model=model,
                val_loader=val_loader,
                val_iters=cfg.val_iters,
            )

            logger.info(
                f'Training iteration {i}: validation loss: {val_stats["loss"]:.4f}, '
                f'validation accuracy: {val_stats["accuracy"]:.2f}%, validation perplexity: {val_stats["perplexity"]:.2f}'
            )

            if tb_writer is not None:
                for k, v in val_stats.items():
                    tb_writer.add_scalar(f'val/{k}', v, i)

            if val_stats['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_stats['accuracy']
                logger.info(f'New best validation accuracy: {val_stats["accuracy"]:.2f}%')
                # save best model
                create_fsdp_full_checkpoint(model, rank, os.path.join(cfg.ckpt_dir, f'{cfg.model_type}-best.pth'))

    # show some training stats.
    logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    fsdp_main()
