# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Train reward model (RM) using QLoRA, starting from a fine-tuned model."""
import os
import itertools
import functools
from typing import Tuple, Mapping, Text, Any
import tqdm
import random
import math
import json
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


from instruct_llama.model_lora import Transformer, LoraModelArgs
from instruct_llama.configs.rm_lora import config as cfg
from instruct_llama.utils.custom_dataset import ComparisonsDataset
from instruct_llama.tokenizer import Tokenizer
from instruct_llama.utils.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.utils.train_helper import (
    create_trace_profiler,
    create_optimizer,
    compute_num_trainable_params,
    get_grad_norm_local,
)
from instruct_llama.utils.logging import create_logger
from instruct_llama.lora import mark_only_lora_as_trainable
from instruct_llama.utils.checkpoint import create_lora_checkpoint, create_normalizer_checkpoint
from instruct_llama.utils.normalizer import RunningMeanStd


logger = create_logger()


def create_checkpoint(model, norm, iter, is_best=False):
    if is_best:
        lora_full_path = os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-best.pth')
        norm_full_path = os.path.join(cfg.ckpt_dir, f'normalizer_{cfg.model_type}-best.pth')
    else:
        lora_full_path = os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{iter}.pth')
        norm_full_path = os.path.join(cfg.ckpt_dir, f'normalizer_{cfg.model_type}-iter-{iter}.pth')

    create_lora_checkpoint(model, lora_full_path, cfg.train_bias, cfg.train_head)
    create_normalizer_checkpoint(norm, norm_full_path)


def clear_gpu_cache(rank=None):
    torch.cuda.empty_cache()


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

    Note we assume the rewards are for the ordered completions for a given prompt,
    where the best completion is the first, and worst completion is the last.
    """
    assert len(rewards.shape) == 1

    N = len(rewards)  # number of responses
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2
    assert C >= 1

    num_accurate = 0

    # for each better completion, compare to the remaining of worse completions
    for i in range(0, N - 1):
        r_rejected = rewards[i + 1 :]
        r_preferred = rewards[i].repeat(len(r_rejected))

        # Perform element-wise comparison
        num_accurate += (r_preferred > r_rejected).sum().item()

    r_best = rewards[0].item()
    r_worst = rewards[-1].item()
    return num_accurate, C, r_best, r_worst


def run_single_train_step(
    model: Transformer,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    reward_stats: RunningMeanStd,
    gradient_accum_steps: int,
    grad_clip: float,
    normalize_reward: bool,
    max_abs_reward: float,
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
    assert gradient_accum_steps >= 1

    metrics = torch.zeros(6).to('cuda')

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    # here one sample is a prompt and a list of completions, where the completions are already ordered by the score, from best to worst
    for batch_tokens, terminal_steps in itertools.islice(train_loader, gradient_accum_steps):
        batch_tokens = batch_tokens.to('cuda', non_blocking=True)
        terminal_steps = terminal_steps.to('cuda', non_blocking=True)

        # forward pass to compute reward for all completions
        outputs = model(batch_tokens).squeeze(-1)  # [num_combinations, seq_length]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [num_combinations]
        raw_rewards = rewards.clone().detach()
        if normalize_reward:
            rewards = reward_stats.normalize(rewards)
        if max_abs_reward > 0:
            rewards = rewards.clamp(min=-max_abs_reward, max=max_abs_reward)

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
        scaled_loss = loss / gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # always update reward norm stats
        reward_stats.update(raw_rewards)

    grad_norm = get_grad_norm_local(model)

    if grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if scaler is not None:  # when using float16
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        optimizer.step()

    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    if return_stats:
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
    model: Transformer,
    val_loader: DataLoader,
    val_iters: int,
    reward_stats: RunningMeanStd,
    normalize_reward: bool,
    max_abs_reward: float,
) -> Mapping[Text, Any]:
    """Run M validation iterations"""

    model.eval()  # set model in validation mode
    metrics = torch.zeros(6).to('cuda')
    inner_pbar = tqdm.tqdm(range(val_iters), colour='green', desc='validation iterations')

    # here one sample is a prompt and a list of completions, where the completions are already ordered by the score, from best to worst
    for batch_tokens, terminal_steps in itertools.islice(val_loader, val_iters):
        batch_tokens = batch_tokens.to('cuda', non_blocking=True)
        terminal_steps = terminal_steps.to('cuda', non_blocking=True)

        # forward pass to compute reward for all completions
        outputs = model(batch_tokens).squeeze(-1)  # [num_combinations, seq_length]

        # get rewards for terminal step, where sequence ends with EOS token, or reached maximum seq_length
        rewards = torch.gather(outputs, dim=1, index=terminal_steps.unsqueeze(-1)).squeeze(1)  # [num_combinations]
        raw_rewards = rewards.clone().detach()
        if normalize_reward:
            rewards = reward_stats.normalize(rewards)
        if max_abs_reward > 0:
            rewards = rewards.clamp(min=-max_abs_reward, max=max_abs_reward)

        # compute loss in a single go
        loss = compute_rm_comparison_loss(rewards)
        num_acc, num_samples, r_best, r_worst = compute_metrics(rewards.detach())
        metrics[0] += loss.detach().item()  # sum up micro batch loss
        metrics[1] += 1  # increase number of samples
        metrics[2] += num_acc  # sum up number of accurate prediction tokens
        metrics[3] += num_samples  # sum up number of responses or combinations
        metrics[4] += r_best  # sum up best reward
        metrics[5] += r_worst  # sum up worst reward

        # # maybe don't update reward norm stats in evaluation mode
        # reward_stats.update(raw_rewards)

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


def init_head_weights(model: Transformer):
    if hasattr(model, 'scalar_head'):
        head = model.scalar_head
        logger.info('Initializing weights for scalar head ...')

        init_std = 1.0 / np.sqrt(model.params.dim + 1)
        torch.nn.init.normal_(head.weight, std=init_std)
        torch.nn.init.zeros_(head.bias)


def main():
    assert cfg.num_epochs >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 1
    assert cfg.val_iters >= 1
    assert cfg.max_abs_reward >= 0.0

    batch_size = int(cfg.gradient_accum_steps)

    assert batch_size >= 1

    if not torch.version.cuda:
        raise RuntimeError('This script requires Pytorch with CUDA.')

    if not os.path.exists(cfg.rm_ckpt_file):
        raise ValueError(f'Invalid model checkpoint "{cfg.rm_ckpt_file}", aborting ...')

    # --------------- Load datasets ---------------

    logger.info('Loading datasets ...')

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
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    num_train_iters = int((len(train_dataset) / batch_size) * cfg.num_epochs)

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = ComparisonsDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)
        val_loader = DataLoader(dataset=val_dataset, collate_fn=_collate_fn, **cuda_kwargs)
        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

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

    model_args = LoraModelArgs.from_model_type(
        model_type=cfg.model_type,
        # LoRA configurations
        lora_r=cfg.lora_r,
        lora_scaling=cfg.lora_scaling,
        lora_dropout=cfg.lora_dropout,
        # LoRA trainable layers
        lora_attn_query=cfg.lora_attn_query,
        lora_attn_key=cfg.lora_attn_key,
        lora_attn_value=cfg.lora_attn_value,
        lora_attn_proj=cfg.lora_attn_proj,
        lora_attn_mlp=cfg.lora_attn_mlp,
        # Quantization configurations
        quant_4bit=cfg.quant_4bit,
        quant_lora_4bit=cfg.quant_lora_4bit,
        quant_4bit_double=cfg.quant_4bit_double,
        quant_4bit_type=cfg.quant_4bit_type,
        quant_compute_dtype=compute_dtype,
        # Regular configurations
        head_type='scalar_head',
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.max_seq_len,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
    )

    model = Transformer(model_args)

    # Load model checkpoint using strict=False,
    # because there's not scalar head weights in the checkpoint state
    if os.path.exists(cfg.rm_ckpt_file):
        logger.info(f'Loading RM checkpoint {cfg.rm_ckpt_file} ...')
        model_state = torch.load(cfg.rm_ckpt_file)
        model.load_state_dict(model_state, strict=False)
        del model_state

    if cfg.random_head_weights:
        model.init_head_weights()

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=compute_dtype)

    mark_only_lora_as_trainable(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

    # This is where the weights quantization happens
    # when we move the model to cuda, the bnb.nn.Params4bit.cuda() method is called,
    # and the weights is quantized using bnb.functional.quantize_4bit
    model = model.to('cuda')

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
        paged_adamw=cfg.use_paged_adamw,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=cfg.init_lr,
        max_lr=cfg.max_lr,
        min_lr=cfg.min_lr,
        warmup_steps=int(cfg.warmup_ratio * num_train_iters),
        max_decay_steps=num_train_iters,
    )

    reward_stats = RunningMeanStd()

    # --------------- Start Training ---------------

    logger.info(
        f'Starting to run {cfg.num_epochs} training epochs, total of {num_train_iters} iterations, with batch size {batch_size}'
    )
    torch_profiler = None
    tb_writer = None
    inner_pbar = None
    best_val_accuracy = 0.0

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    if cfg.use_tensorboard:
        tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    inner_pbar = tqdm.tqdm(range(num_train_iters), colour='blue', desc='Training iterations')

    model.train()

    for i in range(1, num_train_iters + 1):
        train_stats = run_single_train_step(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            reward_stats=reward_stats,
            gradient_accum_steps=cfg.gradient_accum_steps,
            grad_clip=cfg.grad_clip,
            normalize_reward=cfg.normalize_reward,
            max_abs_reward=cfg.max_abs_reward,
            return_stats=i == 1 or i % cfg.log_interval == 0 or i == num_train_iters,
        )

        if inner_pbar is not None:
            inner_pbar.update(1)

        if torch_profiler is not None:
            torch_profiler.step()

        # logging
        if train_stats is not None:
            logger.info(
                f'Training iteration {i}: train loss: {train_stats["loss"]:.4f}, train accuracy: {train_stats["accuracy"]:.2f}%, learning rate: {train_stats["learning_rate"]:.7f}'
            )

            if tb_writer is not None:
                for k, v in train_stats.items():
                    tb_writer.add_scalar(f'train/{k}', v, i)

        # regular checkpointing
        if cfg.ckpt_interval > 0 and (i % cfg.ckpt_interval == 0 or i == num_train_iters):
            create_checkpoint(model, reward_stats, i)

        # validation steps
        if i % cfg.val_interval == 0 or i == num_train_iters:
            val_stats = run_validation_steps(
                model=model,
                val_loader=val_loader,
                val_iters=cfg.val_iters,
                reward_stats=reward_stats,
                normalize_reward=cfg.normalize_reward,
                max_abs_reward=cfg.max_abs_reward,
            )

            logger.info(
                f'Training iteration {i}: validation loss: {val_stats["loss"]:.4f}, validation accuracy: {val_stats["accuracy"]:.2f}%'
            )

            if tb_writer is not None:
                for k, v in val_stats.items():
                    tb_writer.add_scalar(f'val/{k}', v, i)

            if val_stats['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_stats['accuracy']
                logger.info(f'New best validation accuracy: {val_stats["accuracy"]:.2f}%')
                logger.info(f'Reward mean: {reward_stats.mean.item()}, reward variance: {reward_stats.var.item()}')
                # save best model
                create_checkpoint(model, reward_stats, i, is_best=True)

    # training is done ...show some training stats.
    logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
