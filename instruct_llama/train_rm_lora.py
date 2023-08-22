"""Train reward model (RM) using comparison datasets, starting from our fine-tuned model checkpoint, and with LoRA parameter efficient method."""
import os
import itertools
import functools
from typing import Tuple
import tqdm
import random
import math
import numpy as np
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
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
from instruct_llama.utils import ComparisonsDataset
from instruct_llama.lora import lora, lora_state_dict, mark_only_lora_as_trainable

from instruct_llama.configs.train_rm_lora import config as cfg


from instruct_llama.utils import (
    CosineDecayWithWarmupLRScheduler,
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


def compute_rm_comparison_loss(rewards: torch.Tensor) -> torch.Tensor:
    """Compute RM comparison loss.

    Note we assume the rewards are for the ordered completions for a given prompt,
    where the best completion is the first, and worst completion is the last.
    """
    assert len(rewards.shape) == 1  # [num_completions]

    loss = None
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
        r_worser = rewards[i + 1 :]
        r_better = rewards[i].repeat(len(r_worser))
        assert r_better.shape == r_worser.shape

        l_i = -torch.log(torch.sigmoid(r_better - r_worser)).sum()
        if loss is None:
            loss = l_i
        else:
            loss += l_i

    assert loss is not None

    # average over number of combinations
    loss = loss / C

    return loss


@torch.no_grad()
def compute_metrics(rewards: torch.Tensor) -> Tuple[int, int]:
    """Compute number of accurate predictions in terms of reward values.

    Note we assume the rewards are for the ordered completions for a given prompt,
    where the best completion is the first, and worst completion is the last.
    """
    assert len(rewards.shape) == 1  # [num_completions]

    N = len(rewards)  # number of completions
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2
    assert C >= 1

    num_accurate = 0

    # for each better completion, compare to the remaining of worse completions
    for i in range(0, N - 1):
        r_worser = rewards[i + 1 :]
        r_better = rewards[i].repeat(len(r_worser))

        # Perform element-wise comparison
        num_accurate += (r_better > r_worser).sum().item()

    return num_accurate, C


def run_single_train_step(
    model,
    rank,
    world_size,
    train_loader,
    optimizer,
    scheduler,
    scaler=None,
    return_stats=False,
):
    """A single training iteration consists of N micro batch * M gradient accumulation steps.

    ```
    optimizer.zero_grad()
    for step in range(gradient_accum_steps):
        data, target = next(train_loader)
        output = model(data)
        loss = compute_pre_train_loss(output, target)
        loss.backward()

    optimizer.step()
    ```

    """

    local_rank = int(os.environ['LOCAL_RANK'])

    if return_stats:
        metrics = torch.zeros(4).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    # here one sample is a prompt and a list of completions, where the completions are already ordered by the score, from best to worst
    for batch_sequence, terminal_steps in itertools.islice(train_loader, cfg.gradient_accum_steps):
        batch_sequence = batch_sequence.to(local_rank, non_blocking=True)
        terminal_steps = terminal_steps.to(local_rank, non_blocking=True)

        assert len(batch_sequence) == len(terminal_steps)

        B = len(batch_sequence)  # current batch size

        # forward pass to compute reward for all completions
        outputs = []
        for i in range(0, B, cfg.micro_batch_size):
            end_idx = i + cfg.micro_batch_size
            micro_batch_tokens = batch_sequence[i:end_idx]
            outputs.append(model(micro_batch_tokens))

        outputs = torch.concat(outputs, dim=0).to(local_rank).squeeze(-1)  # [batch_size, batch_seq_len]

        # get rewards for terminal step, where sequence ends with EOS token and before the padding tokens
        # from reference:
        # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/rewards.py#L47C48-L47C48
        rewards = torch.gather(outputs, dim=1, index=terminal_steps).squeeze(1)  # [batch_size]

        # compute loss
        loss = compute_rm_comparison_loss(rewards)
        # scale the loss to account for gradient accumulation
        scaled_loss = loss / cfg.gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if return_stats:
            num_acc, num_samples = compute_metrics(rewards.detach())
            metrics[0] += loss.item()  # sum up batch loss
            metrics[1] += 1  # increase number of micro batches
            metrics[2] += num_acc  # sum up number of accurate prediction tokens
            metrics[3] += num_samples  # sum up number of completion

    if scaler is not None:  # when using float16
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)  # unscale before clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    if return_stats:
        train_loss = metrics[0] / metrics[1]
        train_accuracy = 100 * metrics[2] / metrics[3]

        lr = optimizer.param_groups[0]['lr']
        return {
            'loss': train_loss.item(),
            'accuracy': train_accuracy.item(),
            'learning_rate': lr,
        }
    else:
        return None


@torch.no_grad()
def run_validation_steps(model, rank, world_size, val_loader):
    """Run M validation iterations"""

    model.eval()  # set model in validation mode

    local_rank = int(os.environ['LOCAL_RANK'])

    metrics = torch.zeros(4).to(local_rank)

    inner_pbar = tqdm.tqdm(range(cfg.val_iters), colour='green', desc='validation iterations')

    # here one sample is a prompt and a list of completions, where the completions are already ordered by the score, from best to worst
    for batch_sequence, terminal_steps in itertools.islice(val_loader, cfg.val_iters):
        batch_sequence = batch_sequence.to(local_rank, non_blocking=True)
        terminal_steps = terminal_steps.to(local_rank, non_blocking=True)

        assert len(batch_sequence) == len(terminal_steps)

        B = len(batch_sequence)  # current batch size

        # forward pass to compute reward for all completions
        outputs = []
        for i in range(0, B, cfg.micro_batch_size):
            end_idx = i + cfg.micro_batch_size
            micro_batch_tokens = batch_sequence[i:end_idx]
            output = model(micro_batch_tokens)

            outputs.append(output)

        outputs = torch.concat(outputs, dim=0).to(local_rank).squeeze(-1)

        # get rewards for terminal step, where sequence ends with EOS token and before the padding tokens
        rewards = torch.gather(outputs, dim=1, index=terminal_steps).squeeze(1)  # [batch_size]

        # compute loss
        loss = compute_rm_comparison_loss(rewards)
        num_acc, num_samples = compute_metrics(rewards)
        metrics[0] += loss.item()  # sum up batch loss
        metrics[1] += 1  # increase number of batches
        metrics[2] += num_acc  # sum up number of accurate prediction tokens
        metrics[3] += num_samples  # sum up number of completion

        if inner_pbar is not None:
            inner_pbar.update(1)

    val_loss = metrics[0] / metrics[1]
    val_accuracy = 100 * metrics[2] / metrics[3]

    inner_pbar.close()

    model.train()  # set model in training mode after validation runs

    return {'loss': val_loss.item(), 'accuracy': val_accuracy.item()}


def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor]:
    assert len(batch) == 1, 'RM model only support one batch of prompt:completions pair'

    item = batch[0]
    prompt = torch.tensor(item[0], dtype=torch.long)

    # Note we assume the rewards are for the ordered completions for a given prompt,
    # where the best completion is the first, and worst completion is the last.
    completions = item[1]

    max_batch_seq_len = max([len(prompt) + len(completion) for completion in completions])
    assert max_batch_seq_len <= max_seq_len

    if full_pad:
        max_batch_seq_len = max_seq_len

    # concatenate prompt, completion together
    batch_size = len(completions)

    batch_sequences = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)

    # record the terminal index of the completion, often referred to as the terminal time step in RL
    terminal_steps = torch.zeros((batch_size, 1), dtype=torch.long)
    for i, completion in enumerate(completions):
        completion = torch.tensor(completion, dtype=torch.long)
        seq = torch.concat((prompt, completion), dim=0).type(torch.long)
        seq_len = len(seq)

        batch_sequences[i, :seq_len] = seq
        terminal_steps[i] = seq_len - 1  # minus 1 because indexing starts from zero

    return batch_sequences, terminal_steps


def main():
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    if not os.path.exists(cfg.sft_ckpt_file):
        raise ValueError(f'Invalid SFT model checkpoint "{cfg.sft_ckpt_file}", aborting...')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()

    logger = create_logger()

    # --------------- Load datasets ---------------

    logger.info('Loading datasets...')

    tokenizer = Tokenizer(cfg.tokenizer_file)

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=cfg.max_seq_len,
        full_pad=cfg.full_pad,
    )

    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': 1,
        'batch_size': 1,
        'pin_memory': True,
        'shuffle': True,
        'sampler': None,
    }

    train_dataset = ComparisonsDataset(
        data_sources=cfg.train_datasources,
        min_completions=cfg.min_completions,
        max_completions=cfg.max_completions,
        max_seq_len=cfg.max_seq_len,
    )
    train_loader = DataLoader(train_dataset, **cuda_kwargs)
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = ComparisonsDataset(
            data_sources=cfg.val_datasources,
            min_completions=cfg.min_completions,
            max_completions=cfg.max_completions,
            max_seq_len=cfg.max_seq_len,
        )
        val_loader = DataLoader(val_dataset, **cuda_kwargs)
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

        assert model_args.head_type == 'scalar_head'

        model = Transformer(model_args)

        # Load SFT model checkpoint using strict=False,
        # because there's not scalar head weights in the checkpoint state
        if os.path.exists(cfg.sft_ckpt_file):
            logger.info(f'Loading SFT checkpoint {cfg.sft_ckpt_file}...')
            model_state = torch.load(cfg.sft_ckpt_file)
            model.load_state_dict(model_state, strict=False)

            del model_state

    model.init_scalar_head_weights()

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
        logger.warning('Training in float32 mode, make sure you have enough GPU RAM')

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
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        min_lr=cfg.min_lr,
        max_lr=cfg.max_lr,
        warmup_steps=cfg.warmup_steps,
        max_decay_steps=cfg.max_decay_steps,
    )

    # --------------- Start Training ---------------

    logger.info(f'Starting to run {cfg.max_train_iters} training iterations...')

    torch_profiler = None
    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    tb_writer = None
    inner_pbar = None
    train_stats = None
    val_stats = None

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

        inner_pbar = tqdm.tqdm(range(cfg.max_train_iters), colour='blue', desc='Training iterations')

    model.train()
    for iter in range(1, cfg.max_train_iters + 1):
        train_stats = run_single_train_step(
            model=model,
            rank=rank,
            world_size=world_size,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            return_stats=iter % cfg.log_interval == 0 or iter == cfg.max_train_iters,
        )

        if inner_pbar is not None:
            inner_pbar.update(1)

        if torch_profiler is not None:
            torch_profiler.step()

        # logging
        if train_stats is not None and rank == 0:
            logger.info(
                f'Training iteration {iter}: train loss: {train_stats["loss"]:.4f}, train accuracy: {train_stats["accuracy"]:.2f}%, learning rate: {train_stats["learning_rate"]:.10f}'
            )

            if tb_writer is not None:
                tb_writer.add_scalar('train/loss', train_stats['loss'], iter)
                tb_writer.add_scalar('train/accuracy', train_stats['accuracy'], iter)
                tb_writer.add_scalar('train/learning_rate', train_stats['learning_rate'], iter)

        # checkpointing
        if cfg.ckpt_interval > 0 and iter % cfg.ckpt_interval == 0 or iter == cfg.max_train_iters:
            # save model state
            checkpoint = lora_state_dict(model, train_bias=cfg.train_bias, train_head=cfg.train_head)

            torch.save(checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{iter}.pth'))

        # validation steps
        if cfg.val_iters > 0 and (cfg.val_interval > 0 and iter % cfg.val_interval == 0 or iter == cfg.max_train_iters):
            val_stats = run_validation_steps(model=model, rank=rank, world_size=world_size, val_loader=val_loader)

            if rank == 0:
                logger.info(
                    f'Training iteration {iter}: validation loss: {val_stats["loss"]:.4f}, validation accuracy: {val_stats["accuracy"]:.2f}%'
                )

                if tb_writer is not None:
                    tb_writer.add_scalar('val/loss', val_stats['loss'], iter)
                    tb_writer.add_scalar('val/accuracy', val_stats['accuracy'], iter)

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
