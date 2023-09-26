"""Run distributed reward model (RM) training, using comparison datasets, starting from our fine-tuned model checkpoint, and with LoRA parameter efficient method."""
import os
import itertools
import functools
from typing import Tuple, Mapping, Text, Any
import tqdm
import random
import math
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


from instruct_llama.model import RewardModel, TransformerBlock, ModelArgs
from instruct_llama.configs.train_rm_fsdp_lora import config as cfg

from instruct_llama.utils import (
    ComparisonsDataset,
    Tokenizer,
    CosineDecayWithWarmupLRScheduler,
    RunningMeanStd,
    create_trace_profiler,
    create_logger,
    create_optimizer,
    get_grad_norm_local,
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


def compute_rm_comparison_loss(rewards: torch.Tensor) -> torch.Tensor:
    """Compute RM comparison loss for N responses.

    Note we assume the rewards are for the ordered completions for a given prompt,
    where the best completion is the first, and worst completion is the last.
    """
    assert len(rewards.shape) == 1  # [num_completions]

    losses = None
    N = len(rewards)  # number of completions
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2
    assert C >= 1

    # for each better completion 0, 1, ..., N-1, compare to the remaining of worse completions
    # for example:
    # 0 <-> (1, 2, ..., N)
    # 1 <-> (2, 3, ..., N)
    # N-1 <-> (N)
    for i in range(0, N - 1):
        r_rejected = rewards[i + 1 :]
        r_preferred = rewards[i].repeat(len(r_rejected))
        assert r_preferred.shape == r_rejected.shape

        loss = -torch.nn.functional.logsigmoid(r_preferred - r_rejected).sum()
        if losses is None:
            losses = loss
        else:
            losses += loss

    assert losses is not None

    # average over number of combinations
    loss = losses / C

    return loss


@torch.no_grad()
def compute_metrics(rewards: torch.Tensor) -> Tuple[int, int, float, float]:
    """Compute number of accurate predictions in terms of reward values.

    Note we assume the rewards are for the ordered responses for a given prompt,
    where the best completion is the first, and worst completion is the last.
    """
    assert len(rewards.shape) == 1

    N = len(rewards)  # number of responses
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2
    assert C >= 1

    num_accurate = 0

    # for each better completion, compare to the remaining of worse responses
    for i in range(0, N - 1):
        r_rejected = rewards[i + 1 :]
        r_preferred = rewards[i].repeat(len(r_rejected))

        # Perform element-wise comparison
        num_accurate += (r_preferred > r_rejected).sum().item()

    r_best = rewards[0].item()
    r_worst = rewards[-1].item()
    return num_accurate, C, r_best, r_worst


def run_single_train_step(
    model: RewardModel,
    rank: int,
    world_size: int,
    local_rank: int,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    reward_stats: RunningMeanStd,
    return_stats: bool = False,
) -> Mapping[Text, Any]:
    """A single training iteration consists of N micro batch * M gradient accumulation steps.

    ```
    optimizer.zero_grad()
    for step in range(gradient_accum_steps):
        data, target = next(train_loader)
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()

    optimizer.step()
    ```

    """

    metrics = torch.zeros(6).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    # here one sample is a prompt and a list of completions, where the completions are already ordered by the score, from best to worst
    for batch_tokens, terminal_steps in itertools.islice(train_loader, cfg.gradient_accum_steps):
        batch_tokens = batch_tokens.to(local_rank, non_blocking=True)
        terminal_steps = terminal_steps.to(local_rank, non_blocking=True)

        # # forward pass to compute reward for all completions
        outputs = model(batch_tokens).squeeze(-1)  # [num_combinations, seq_length]
        # rewards = outputs[:, -1]  # [num_combinations]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [num_combinations]

        if cfg.normalize_reward:
            rewards = reward_stats.normalize(rewards)

        # compute loss in a single go
        loss = compute_rm_comparison_loss(rewards)
        num_acc, num_samples, r_best, r_worst = compute_metrics(rewards.detach())
        metrics[0] += loss.detach().item()  # sum up micro batch loss
        metrics[1] += 1  # increase number of samples
        metrics[2] += num_acc  # sum up number of accurate prediction tokens
        metrics[3] += num_samples  # sum up number of responses or combinations
        metrics[4] += r_best  # sum up best reward
        metrics[5] += r_worst  # sum up worst reward

        # scale the loss to account for gradient accumulation
        scaled_loss = loss / cfg.gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # always update reward norm stats
        reward_stats.update(rewards.detach())

    grad_norm = get_grad_norm_local(model)

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
        loss_mean = metrics[0] / metrics[1]
        accuracy = 100 * metrics[2] / metrics[3]
        preferred_reward_mean = metrics[4] / metrics[1]
        rejected_reward_mean = metrics[5] / metrics[1]

        return {
            'loss': loss_mean.item(),
            'accuracy': accuracy.item(),
            'preferred_reward': preferred_reward_mean.item(),
            'rejected_reward': rejected_reward_mean.item(),
            'reward_gap': (preferred_reward_mean - rejected_reward_mean).item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm.item(),
        }
    else:
        return None


@torch.no_grad()
def run_validation_steps(
    model: RewardModel,
    rank: int,
    world_size: int,
    local_rank: int,
    val_loader: DataLoader,
    reward_stats: RunningMeanStd,
) -> Mapping[Text, Any]:
    """Run M validation iterations"""

    model.eval()  # set model in validation mode
    metrics = torch.zeros(6).to(local_rank)
    inner_pbar = tqdm.tqdm(range(cfg.val_iters), colour='green', desc='validation iterations')

    # here one sample is a prompt and a list of completions, where the completions are already ordered by the score, from best to worst
    for batch_tokens, terminal_steps in itertools.islice(val_loader, cfg.val_iters):
        batch_tokens = batch_tokens.to(local_rank, non_blocking=True)
        terminal_steps = terminal_steps.to(local_rank, non_blocking=True)

        # forward pass to compute reward for all completions
        outputs = model(batch_tokens).squeeze(-1)  # [num_combinations, seq_length]
        # rewards = outputs[:, -1]  # [num_combinations]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [num_combinations]

        if cfg.normalize_reward:
            rewards = reward_stats.normalize(rewards)

        # compute loss in a single go
        loss = compute_rm_comparison_loss(rewards)
        num_acc, num_samples, r_best, r_worst = compute_metrics(rewards.detach())
        metrics[0] += loss.detach().item()  # sum up micro batch loss
        metrics[1] += 1  # increase number of samples
        metrics[2] += num_acc  # sum up number of accurate prediction tokens
        metrics[3] += num_samples  # sum up number of responses or combinations
        metrics[4] += r_best  # sum up best reward
        metrics[5] += r_worst  # sum up worst reward

        # always update reward norm stats
        reward_stats.update(rewards.detach())

        if inner_pbar is not None:
            inner_pbar.update(1)

    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    loss = metrics[0] / metrics[1]
    accuracy = 100 * metrics[2] / metrics[3]
    preferred_reward_mean = metrics[4] / metrics[1]
    rejected_reward_mean = metrics[5] / metrics[1]

    inner_pbar.close()

    model.train()  # set model in training mode after validation runs

    return {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'preferred_reward': preferred_reward_mean.item(),
        'rejected_reward': rejected_reward_mean.item(),
        'reward_gap': (preferred_reward_mean - rejected_reward_mean).item(),
    }


def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(batch) == 1, 'This script only support one item at a time to validate RM model'
    tokens_list = batch[0]

    # Note we assume the tokens are for the ordered completions for a given prompt,
    # where the best completion is the first, and worst completion is the last.

    max_batch_seq_len = max([len(tokens) for tokens in tokens_list])
    assert max_batch_seq_len <= max_seq_len

    if full_pad:
        max_batch_seq_len = max_seq_len

    # concatenate prompt, completion together
    batch_size = len(tokens_list)

    batch_sequences = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)

    # record the terminal index of the completion, often referred to as the terminal time step in RL
    terminal_steps = torch.zeros((batch_size), dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        seq = torch.tensor(tokens, dtype=torch.long)
        seq_len = len(seq)

        batch_sequences[i, :seq_len] = seq
        terminal_steps[i] = seq_len - 1  # minus 1 because indexing starts from zero

    return batch_sequences, terminal_steps


def rank0_logger(msg, rank):
    if rank == 0:
        print(msg)


def main():
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 1
    assert cfg.val_iters >= 1

    if not os.path.exists(cfg.rm_ckpt_file):
        raise ValueError(f'Invalid model checkpoint "{cfg.rm_ckpt_file}", aborting...')

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
        'num_workers': 1,
        'batch_size': 1,  # always work on one sample at a time
        'pin_memory': True,
        'shuffle': True,
        'sampler': None,
    }

    train_dataset = ComparisonsDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=_collate_fn, **cuda_kwargs)
    logger(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = ComparisonsDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)
        val_loader = DataLoader(dataset=val_dataset, collate_fn=_collate_fn, **cuda_kwargs)
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
        model_args.resid_dropout = cfg.resid_dropout
        model_args.head_type = cfg.head_type
        model = RewardModel(model_args)

        # Load model checkpoint using strict=False,
        # because there's not scalar head weights in the checkpoint state
        if os.path.exists(cfg.rm_ckpt_file):
            logger(f'Loading RM checkpoint {cfg.rm_ckpt_file}...')
            model_state = torch.load(cfg.rm_ckpt_file)
            model.load_state_dict(model_state, strict=False)
            del model_state

    if cfg.random_head_weights:
        model.init_head_weights()

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

    reward_stats = RunningMeanStd()

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
            reward_stats=reward_stats,
            return_stats=i == 1 or i % cfg.log_interval == 0 or i == cfg.max_train_iters,
        )

        if inner_pbar is not None:
            inner_pbar.update(1)

        if torch_profiler is not None:
            torch_profiler.step()

        # logging
        if train_stats is not None and rank == 0:
            logger(
                f'Training iteration {i}: train loss: {train_stats["loss"]:.4f}, train accuracy: {train_stats["accuracy"]:.2f}%, learning rate: {train_stats["learning_rate"]:.10f}'
            )

            if tb_writer is not None:
                for k, v in train_stats.items():
                    tb_writer.add_scalar(f'train/{k}', v, i)

        # validation steps
        if i % cfg.val_interval == 0 or i == cfg.max_train_iters:
            val_stats = run_validation_steps(
                model=model,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                val_loader=val_loader,
                reward_stats=reward_stats,
            )

            if rank == 0:
                logger(
                    f'Training iteration {i}: validation loss: {val_stats["loss"]:.4f}, validation accuracy: {val_stats["accuracy"]:.2f}%'
                )

                if tb_writer is not None:
                    for k, v in val_stats.items():
                        tb_writer.add_scalar(f'val/{k}', v, i)

            # checkpointing
            if val_stats['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_stats['accuracy']
                logger(f'New best validation accuracy: {val_stats["accuracy"]:.2f}%')
                # save model state
                save_lora_model_checkpoint(
                    model=model,
                    rank=rank,
                    ckpt_save_path=os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{i}.pth'),
                    train_bias=cfg.train_bias,
                    train_head=cfg.train_head,
                )

    # save final model state
    if rank == 0:
        logger(f'Reward mean: {reward_stats.mean.item()}, reward variance: {reward_stats.var.item()}')
    save_lora_model_checkpoint(
        model=model,
        rank=rank,
        ckpt_save_path=os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{cfg.max_train_iters}.pth'),
        train_bias=cfg.train_bias,
        train_head=cfg.train_head,
    )

    if rank == 0:
        # training is done...show some training stats.
        logger(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
