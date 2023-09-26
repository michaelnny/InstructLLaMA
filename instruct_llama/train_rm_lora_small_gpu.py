"""Train reward model (RM) using comparison datasets, starting from our fine-tuned model checkpoint, and with LoRA parameter efficient method."""
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


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import RewardModel, ModelArgs
from instruct_llama.configs.train_rm_lora import config as cfg

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


def compute_rm_comparison_loss_for_single_pair(rewards: torch.Tensor) -> torch.Tensor:
    """Compute RM comparison loss."""

    assert len(rewards) == 2

    r_preferred = rewards[0]
    r_rejected = rewards[1]

    loss = -torch.nn.functional.logsigmoid(r_preferred - r_rejected)

    return loss


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
    losses = losses / C

    return losses


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
        data, target = next(iter(train_loader))
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()

    optimizer.step()
    ```

    """

    if return_stats:
        metrics = torch.zeros(6).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    # note we should be able to use similar procedure within the 'run_validation_steps' if we have more powerful GPUs
    for single_sample in itertools.islice(train_loader, cfg.gradient_accum_steps):  # for each training sample
        C = len(single_sample)  # number of combinations
        micro_batch_rewards = []
        micro_batch_loss = 0
        for batch_tokens, terminal_steps in single_sample:  # for each combination
            batch_tokens = batch_tokens.to(local_rank, non_blocking=True)
            terminal_steps = terminal_steps.to(local_rank, non_blocking=True)

            # forward pass to compute reward
            outputs = model(batch_tokens).squeeze(-1)  # [2, seq_len]

            # get rewards for terminal step, which is the first EOS token in the completion
            # from reference:
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/rewards.py#L47C48-L47C48
            rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(1)).squeeze(1)  # [2]

            if cfg.normalize_reward:
                rewards = reward_stats.normalize(rewards)

            # compute loss
            loss = compute_rm_comparison_loss_for_single_pair(rewards)
            loss *= 1 / C  # scale by number of combinations

            # scale the loss to account for gradient accumulation
            scaled_loss = loss / cfg.gradient_accum_steps

            if scaler is not None:  # when using float16
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # always update reward norm stats
            reward_stats.update(rewards.detach())

            # for logging
            micro_batch_rewards.append(rewards.detach())
            micro_batch_loss += loss.detach()

        if return_stats:
            micro_batch_rewards = torch.concat(micro_batch_rewards)
            num_acc, num_samples, r_best, r_worst = compute_metrics(micro_batch_rewards)
            metrics[0] += micro_batch_loss.item()  # sum up micro batch loss
            metrics[1] += 1  # increase number of samples
            metrics[2] += num_acc  # sum up number of accurate prediction tokens
            metrics[3] += num_samples  # sum up number of responses or combinations
            metrics[4] += r_best  # sum up best reward
            metrics[5] += r_worst  # sum up worst reward

    grad_norm = get_grad_norm_local(model)

    if cfg.grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

    if scaler is not None:  # when using float16
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        optimizer.step()

    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    if return_stats:
        loss = metrics[0] / metrics[1]
        accuracy = 100 * metrics[2] / metrics[3]
        preferred_reward_mean = metrics[4] / metrics[1]
        rejected_reward_mean = metrics[5] / metrics[1]

        return {
            'loss': loss.item(),
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

        # forward pass to compute reward for all completions, since we're in no grad mode
        outputs = model(batch_tokens).squeeze(-1)  # [num_combinations, seq_len]

        # get rewards for terminal step, where sequence ends with EOS token and before the padding tokens
        rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [num_combinations]

        if cfg.normalize_reward:
            rewards = reward_stats.normalize(rewards)

        # compute loss in a single go
        loss = compute_rm_comparison_loss(rewards)
        num_acc, num_samples, r_best, r_worst = compute_metrics(rewards)
        metrics[0] += loss.item()  # sum up micro batch loss
        metrics[1] += 1  # increase number of samples
        metrics[2] += num_acc  # sum up number of accurate prediction tokens
        metrics[3] += num_samples  # sum up number of responses or combinations
        metrics[4] += r_best  # sum up best reward
        metrics[5] += r_worst  # sum up worst reward

        # always update reward norm stats
        reward_stats.update(rewards.detach())

        if inner_pbar is not None:
            inner_pbar.update(1)

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


def custom_collate_fn_train(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(batch) == 1, 'This script only support one item at a time to train RM model'
    tokens_list = batch[0]

    # Note we assume the tokens are for the ordered completions for a given prompt,
    # where the best completion is the first, and worst completion is the last.

    max_batch_seq_len = max([len(tokens) for tokens in tokens_list])
    assert max_batch_seq_len <= max_seq_len

    # however due to limited GPU resource, we can't compute all these in a single forward and backward pass,
    # thus we have to break then into micro batches, which means increased computations

    N = len(tokens_list)  # number of responses
    C = math.comb(N, 2)  # number of combinations

    micro_batches = []

    for i in range(0, N - 1):
        preferred = tokens_list[i]
        for j in range(i + 1, N):
            rejected = tokens_list[j]
            max_batch_seq_len = max(len(preferred), len(rejected))
            if full_pad:
                max_batch_seq_len = max_seq_len
            batch_sequences = torch.full((2, max_batch_seq_len), pad_id, dtype=torch.long)
            terminal_steps = torch.zeros((2), dtype=torch.long)

            batch_sequences[0, : len(preferred)] = torch.tensor(preferred, dtype=torch.long)
            batch_sequences[1, : len(rejected)] = torch.tensor(rejected, dtype=torch.long)

            terminal_steps[0] = len(preferred) - 1
            terminal_steps[1] = len(rejected) - 1

            micro_batches.append((batch_sequences, terminal_steps))

    assert len(micro_batches) == C
    return micro_batches


def custom_collate_fn_val(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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

    logger = create_logger()

    # --------------- Load datasets ---------------

    logger.info('Loading datasets...')

    tokenizer = Tokenizer(cfg.tokenizer_file)

    # we have to use a inefficient collate function for training,
    # since we only have a single GPU and we can't fit the entire N comparisons in a single forward pass
    _collate_fn_train = functools.partial(
        custom_collate_fn_train,
        pad_id=tokenizer.eos_id,
        max_seq_len=cfg.max_seq_len,
        full_pad=cfg.full_pad,
    )
    _collate_fn_val = functools.partial(
        custom_collate_fn_val,
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
    train_loader = DataLoader(dataset=train_dataset, collate_fn=_collate_fn_train, **cuda_kwargs)
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = ComparisonsDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)
        val_loader = DataLoader(dataset=val_dataset, collate_fn=_collate_fn_val, **cuda_kwargs)
        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    logger.info('Initialize model and optimizer...')

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
            logger.info(f'Loading RM checkpoint {cfg.rm_ckpt_file}...')
            model_state = torch.load(cfg.rm_ckpt_file)
            model.load_state_dict(model_state, strict=False)
            del model_state

    if cfg.random_head_weights:
        model.init_head_weights()

    mark_only_lora_as_trainable(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported()
    train_dtype = torch.float32

    scaler = None

    if cfg.mixed_precision:
        if bf16_ready:
            train_dtype = torch.bfloat16
        else:
            train_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    else:
        logger.info('Training in float32 mode, make sure you have enough GPU RAM')

    # BUG in pytorch 2.0.1, as we found out using torch.autocast will increase GPU RAM usage,
    # and cause CUDA OUT OF MEMORY error when run the training script on a single RTX 3090
    # so we manually convert the model to half precision before moving it to GPU

    # mp_ctx = torch.cuda.amp.autocast(dtype=train_dtype, cache_enabled=False)

    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=train_dtype)

    model = model.to(local_rank)

    if cfg.compile_model:
        logger.info('compile model using torch.compile()...')
        model = torch.compile(model)

    logger.info('Initialize optimizer...')

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

    logger.info(f'Starting to run {cfg.max_train_iters} training iterations...')

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
            logger.info(
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
                logger.info(
                    f'Training iteration {i}: validation loss: {val_stats["loss"]:.4f}, validation accuracy: {val_stats["accuracy"]:.2f}%'
                )

                if tb_writer is not None:
                    for k, v in val_stats.items():
                        tb_writer.add_scalar(f'val/{k}', v, i)

            # checkpointing
            if val_stats['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_stats['accuracy']
                logger.info(f'New best validation accuracy: {val_stats["accuracy"]:.2f}%')
                logger.info(f'Reward mean: {reward_stats.mean.item()}, reward variance: {reward_stats.var.item()}')
                # save model state
                checkpoint = lora_state_dict(model, train_bias=cfg.train_bias, train_head=cfg.train_head)
                torch.save(checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{i}.pth'))

    # save final model state
    logger.info(f'Reward mean: {reward_stats.mean.item()}, reward variance: {reward_stats.var.item()}')
    torch.save(checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{cfg.max_train_iters}.pth'))

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
