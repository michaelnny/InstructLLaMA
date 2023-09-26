"""Run distributed SFT, starting with Meta's pretrained model."""
import os
import itertools
import functools
from typing import Tuple, Mapping, Text, Any
import tqdm
import random
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
from instruct_llama.configs.train_sft_fsdp_lora import config as cfg

from instruct_llama.utils import (
    FineTuneDataset,
    Tokenizer,
    CosineDecayWithWarmupLRScheduler,
    create_trace_profiler,
    create_optimizer,
    get_grad_norm_fsdp,
    lora,
    mark_only_lora_as_trainable,
    save_lora_model_checkpoint,
)


def setup():
    # initialize the process group
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f'clearing cache for rank {rank}')
    torch.cuda.empty_cache()


# FSDP activation checkpointing
non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    check_fn = lambda submodule: isinstance(submodule, TransformerBlock)  # noqa: E731

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


def compute_finetune_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == len(mask.shape) == 2  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0] == mask.shape[0]

    B, T, *_ = logits.shape

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')

    assert not torch.any(torch.isnan(loss))

    loss = loss.view(B, T)

    assert loss.shape == mask.shape

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # note here prompt is less important than completion
    weights = mask.float().masked_fill(mask == -1, cfg.prompt_loss_weight).masked_fill(mask == 1, cfg.completion_loss_weight)
    loss *= weights

    loss = torch.mean(loss)

    return loss


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_len]
    assert targets.shape == mask.shape  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0]

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # only include completion when compute accuracy
    weights = mask.float().masked_fill(mask == -1, 0)

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)

    correct = pred.eq(targets.view_as(pred)).float()

    # only consider completion when compute metrics
    correct *= weights
    num_accurate = correct.sum().item()
    num_samples = weights.bool().sum().item()

    return (num_accurate, num_samples)


def run_single_train_step(
    model: Transformer,
    rank: int,
    world_size: int,
    local_rank: int,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    scaler: torch.cuda.amp.GradScaler,
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

    metrics = torch.zeros(5).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    for x, y, loss_mask in itertools.islice(train_loader, cfg.gradient_accum_steps):
        x, y, loss_mask = (
            x.to(local_rank, non_blocking=True),
            y.to(local_rank, non_blocking=True),
            loss_mask.to(local_rank, non_blocking=True),
        )

        output = model(x)

        loss = compute_finetune_loss(output, y, loss_mask)
        # scale the loss to account for gradient accumulation
        scaled_loss = loss / cfg.gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        num_acc, num_samples = compute_metrics(output, y, loss_mask)
        metrics[0] += loss.item()  # sum up batch loss
        metrics[1] += np.exp(loss.item())  # sum up perplexity
        metrics[2] += 1  # increase number of micro batches
        metrics[3] += num_acc  # sum up number of accurate prediction tokens
        metrics[4] += num_samples  # sum up number of tokens

    grad_norm = get_grad_norm_fsdp(model, rank, world_size, cfg.sharding_strategy)

    if cfg.grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        #  FSDP needs to use this method to clip gradient norm instead of torch.nn.utils.clip_grad_norm_
        model.clip_grad_norm_(cfg.grad_clip)

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
    model: Transformer, rank: int, world_size: int, local_rank: int, val_loader: DataLoader
) -> Mapping[Text, Any]:
    """Run M validation iterations"""
    model.eval()  # set model in validation mode

    metrics = torch.zeros(5).to(local_rank)

    inner_pbar = tqdm.tqdm(range(cfg.val_iters), colour='green', desc='validation iterations')

    for x, y, loss_mask in itertools.islice(val_loader, cfg.val_iters):
        x, y, loss_mask = (
            x.to(local_rank, non_blocking=True),
            y.to(local_rank, non_blocking=True),
            loss_mask.to(local_rank, non_blocking=True),
        )

        output = model(x)

        loss = compute_finetune_loss(output, y, loss_mask)
        num_acc, num_samples = compute_metrics(output, y, loss_mask)
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


def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor]:
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    and compute the loss mask for the batch.
    """

    batch_size = len(batch)

    max_batch_seq_len = max([len(item[0]) + len(item[1]) for item in batch])
    assert max_batch_seq_len <= max_seq_len

    if full_pad:
        max_batch_seq_len = max_seq_len

    # concatenate prompt, completion together
    batch_sequences = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)

    # loss mask where -1s are prompt tokens, 1s are completion tokens, and 0s are padding tokens
    loss_mask = torch.full((batch_size, max_batch_seq_len), 0, dtype=torch.long)

    for i, (prompt, completion) in enumerate(batch):
        # need prompt, completion lengths to compute loss mask
        prompt_len, completion_len = len(prompt), len(completion)
        seq_len = prompt_len + completion_len
        seq = torch.concat((prompt, completion), dim=0).type(torch.long)

        # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
        batch_sequences[i, :seq_len] = seq
        loss_mask[i, :prompt_len] = -1  # prompt tokens
        loss_mask[i, prompt_len : prompt_len + completion_len] = 1  # completion tokens

    x = batch_sequences[:, :-1]  # [batch_size, max_batch_seq_len - 1]
    y = batch_sequences[:, 1:]  # [batch_size, max_batch_seq_len - 1]

    # shift to right to align with y
    loss_mask = loss_mask[:, 1:]

    return x, y, loss_mask


def rank0_logger(msg, rank):
    if rank == 0:
        print(msg)


def main():
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint "{cfg.pretrain_ckpt_file}", aborting...')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()

    logger = functools.partial(rank0_logger, rank=rank)

    # --------------- Load datasets ---------------

    logger('Loading datasets...')

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
        'pin_memory': True,
        'shuffle': False,
    }

    train_dataset = FineTuneDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    train_kwargs = {'batch_size': cfg.micro_batch_size, 'sampler': train_sampler}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    logger(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = FineTuneDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)
        val_kwargs = {'batch_size': cfg.val_batch_size, 'sampler': None}
        val_kwargs.update(cuda_kwargs)
        val_loader = DataLoader(val_dataset, **val_kwargs)
        logger(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    logger('Initialize model and optimizer...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    with lora(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, enabled=True):
        model_args = ModelArgs.from_model_type(cfg.model_type)
        model_args.vocab_size = tokenizer.vocab_size
        model_args.max_seq_len = cfg.max_seq_len
        model_args.embed_dropout = cfg.embed_dropout
        model_args.attn_dropout = cfg.attn_dropout
        model_args.head_type = cfg.head_type

        assert model_args.head_type == 'lm_head'

        model = Transformer(model_args)

        # Load model checkpoint using strict=False,
        # because there are missing keys due to LoRA weights not contained in checkpoint state
        if os.path.exists(cfg.pretrain_ckpt_file):
            logger(f'Loading pretrained checkpoint {cfg.pretrain_ckpt_file}...')
            model_state = torch.load(cfg.pretrain_ckpt_file)
            model.load_state_dict(model_state, strict=False)

            del model_state

    mark_only_lora_as_trainable(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    scaler = None
    mixed_precision_policy = None  # defaults to fp32

    if cfg.mixed_precision:
        bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported() and dist.is_nccl_available()
        if bf16_ready:
            logger('--> bFloat16 enabled for mixed precision...')
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )
        else:
            logger('--> float16 enabled for mixed precision...')
            # requires grad scaler
            scaler = ShardedGradScaler()
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                # Gradient communication precision.
                reduce_dtype=torch.float16,
                # Buffer precision.
                buffer_dtype=torch.float16,
            )
    else:
        logger('--> fallback to float32...')

    _auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})

    model = FSDP(
        model,
        auto_wrap_policy=_auto_wrap_policy,
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
        logger('--> applying FSDP activation checkpointing...')
        apply_fsdp_checkpointing(model)

    logger(f'--> FSDP model:\n{model}')

    if cfg.compile_model:
        logger('compile model using torch.compile()...')
        model = torch.compile(model)

    logger('Initialize optimizer...')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
        use_bnb_8bit=cfg.use_bnb_8bit,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=cfg.init_lr,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps,
        max_decay_steps=cfg.max_decay_steps,
    )

    # --------------- Start Training ---------------

    logger(f'Starting to run {cfg.max_train_iters} training iterations...')

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

        inner_pbar = tqdm.tqdm(range(cfg.max_train_iters), colour='blue', desc='Training iterations')

    model.train()

    for i in range(1, cfg.max_train_iters + 1):
        train_stats = run_single_train_step(
            model=model,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            return_stats=i == 1 or i % cfg.log_interval == 0 or i == cfg.max_train_iters,
        )

        if inner_pbar is not None:
            inner_pbar.update(1)

        if torch_profiler is not None:
            torch_profiler.step()

        # logging
        if train_stats is not None and rank == 0:
            logger(
                f'Training iteration {i}: train loss: {train_stats["loss"]:.4f}, '
                f'train accuracy: {train_stats["accuracy"]:.2f}%, train perplexity: {train_stats["perplexity"]:.2f}, learning rate: {train_stats["learning_rate"]:.10f}'
            )

            if tb_writer is not None:
                for k, v in train_stats.items():
                    tb_writer.add_scalar(f'train/{k}', v, i)

        # validation steps
        if cfg.val_iters > 0 and (cfg.val_interval > 0 and i % cfg.val_interval == 0 or i == cfg.max_train_iters):
            val_stats = run_validation_steps(
                model=model, rank=rank, world_size=world_size, local_rank=local_rank, val_loader=val_loader
            )

            if rank == 0:
                logger(
                    f'Training iteration {i}: validation loss: {val_stats["loss"]:.4f}, '
                    f'validation accuracy: {val_stats["accuracy"]:.2f}%, validation perplexity: {val_stats["perplexity"]:.2f}'
                )

                if tb_writer is not None:
                    for k, v in val_stats.items():
                        tb_writer.add_scalar(f'val/{k}', v, i)

            # checkpointing
            if val_stats['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_stats['accuracy']
                logger.info(f'New best validation accuracy: {val_stats["accuracy"]:.2f}%')
                # save model state
                save_lora_model_checkpoint(
                    model=model,
                    rank=rank,
                    ckpt_save_path=os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{i}.pth'),
                    train_bias=cfg.train_bias,
                    train_head=cfg.train_head,
                )

    # ------------ training is done ------------
    save_lora_model_checkpoint(
        model=model,
        rank=rank,
        ckpt_save_path=os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{cfg.max_train_iters}.pth'),
        train_bias=cfg.train_bias,
        train_head=cfg.train_head,
    )

    if rank == 0:
        # show some training stats.
        print(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
