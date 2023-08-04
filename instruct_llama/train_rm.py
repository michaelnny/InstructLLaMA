"""Train reward model (RM) starting from our fine-tuned model."""
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

from instruct_llama.configs.train_rm import config as cfg


from instruct_llama.utils import (
    CosineDecayWithWarmupLRScheduler,
    Memory_Maximizer,
    format_to_gb,
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
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )

    return torch_profiler


def compute_rm_comparison_loss(rewards: torch.Tensor) -> torch.Tensor:
    assert len(rewards.shape) == 1  # [num_completions]

    loss = None
    N = len(rewards)  # number of completions
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2
    assert C >= 1

    for i in range(0, N - 1):
        r_better = rewards[i]
        for j in range(i + 1, N):
            r_worser = rewards[j]

            if loss is None:
                loss = (1 / C) * -torch.log(torch.sigmoid(r_better - r_worser))
            else:
                loss += (1 / C) * -torch.log(torch.sigmoid(r_better - r_worser))

    return loss


@torch.no_grad()
def compute_metrics(rewards: torch.Tensor) -> Tuple[int, int]:
    assert len(rewards.shape) == 1  # [num_completions]

    N = len(rewards)  # number of completions
    C = math.comb(N, 2)  # number of combinations

    assert N >= 2
    assert C >= 1

    num_accurate = 0

    for i in range(0, N - 1):
        r_better = rewards[i]
        for j in range(i + 1, N):
            r_worser = rewards[j]

            if r_better > r_worser:
                num_accurate += 1

    return num_accurate, C


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


def run_single_train_step(
    ctx,
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
        fsdp_metrics = torch.zeros(4).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    for x, ys in itertools.islice(train_loader, cfg.gradient_accum_steps):
        # forward pass to compute reward for each completion
        rewards = torch.zeros((len(ys))).to(local_rank)

        x = torch.tensor(x, dtype=torch.long).to(local_rank)
        for i, y in enumerate(ys):
            y = torch.tensor(y, dtype=torch.long).to(local_rank)

            tokens = torch.concat((x, y), dim=0).type(torch.long).to(local_rank)
            tokens = tokens.unsqueeze(0)
            with ctx:
                output = model(tokens)  # [1, seq_len, 1]

                # only use the reward from terminal time step
                r_T = output.squeeze()[-1]
                rewards[i] = r_T

        # compute loss
        loss = compute_rm_comparison_loss(rewards)
        # scale the loss to account for gradient accumulation, we assume average completions per prompt is 4
        scaled_loss = loss / (cfg.gradient_accum_steps * 4)

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if return_stats:
            num_acc, num_samples = compute_metrics(rewards)
            fsdp_metrics[0] += loss.item()  # sum up batch loss
            fsdp_metrics[1] += 1  # increase number of batches
            fsdp_metrics[2] += num_acc  # sum up number of accurate prediction tokens
            fsdp_metrics[3] += num_samples  # sum up number of completion

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
        train_loss = fsdp_metrics[0] / fsdp_metrics[1]
        train_accuracy = 100 * fsdp_metrics[2] / fsdp_metrics[3]

        lr = optimizer.param_groups[0]['lr']
        return {
            'loss': train_loss.item(),
            'accuracy': train_accuracy.item(),
            'learning_rate': lr,
        }
    else:
        return None


def run_validation_steps(ctx, model, rank, world_size, val_loader):
    """Run M validation iterations"""

    model.eval()  # set model in validation mode

    local_rank = int(os.environ['LOCAL_RANK'])

    fsdp_metrics = torch.zeros(4).to(local_rank)

    inner_pbar = tqdm.tqdm(range(cfg.val_iters), colour='green', desc='validation iterations')

    with torch.no_grad():
        for x, ys in itertools.islice(val_loader, cfg.val_iters):
            # forward pass to compute reward for each completion
            rewards = torch.zeros((len(ys))).to(local_rank)

            x = torch.tensor(x, dtype=torch.long).to(local_rank)
            for i, y in enumerate(ys):
                y = torch.tensor(y, dtype=torch.long).to(local_rank)

                tokens = torch.concat((x, y), dim=0).type(torch.long).to(local_rank)
                tokens = tokens.unsqueeze(0)
                with ctx:
                    output = model(tokens)  # [1, seq_len, 1]

                    # only use the reward from terminal time step
                    r_T = output.squeeze()[-1]
                    rewards[i] = r_T

            # compute loss
            loss = compute_rm_comparison_loss(rewards)
            num_acc, num_samples = compute_metrics(rewards)
            fsdp_metrics[0] += loss.item()  # sum up batch loss
            fsdp_metrics[1] += 1  # increase number of batches
            fsdp_metrics[2] += num_acc  # sum up number of accurate prediction tokens
            fsdp_metrics[3] += num_samples  # sum up number of completion

    if inner_pbar is not None:
        inner_pbar.update(1)

    val_loss = fsdp_metrics[0] / fsdp_metrics[1]
    val_accuracy = 100 * fsdp_metrics[2] / fsdp_metrics[3]

    inner_pbar.close()

    model.train()  # set model in training mode after validation runs

    return {'loss': val_loss.item(), 'accuracy': val_accuracy.item()}


def main():
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint "{cfg.pretrain_ckpt_file}", aborting...')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()

    logger = create_logger()

    # --------------- Load datasets ---------------

    logger.info('Loading datasets ...')

    tokenizer = Tokenizer(cfg.tokenizer_file)

    train_dataset = ComparisonsDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)

    cuda_kwargs = {
        'num_workers': 1,
        'batch_size': 1,
        'pin_memory': True,
        'shuffle': True,
        'sampler': None,
    }

    train_loader = DataLoader(train_dataset, **cuda_kwargs)

    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = ComparisonsDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)

        val_loader = DataLoader(val_dataset, **cuda_kwargs)

        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    logger.info('Initialize model and optimizer ...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    model_args = ModelArgs.from_model_type(cfg.model_type)
    model_args.vocab_size = tokenizer.vocab_size
    model_args.max_seq_len = cfg.max_seq_len
    model_args.embed_dropout = cfg.embed_dropout
    model_args.attn_dropout = cfg.attn_dropout
    model_args.resid_dropout = cfg.resid_dropout
    model_args.head_type = cfg.head_type

    assert model_args.head_type == 'scalar_head'

    model = Transformer(model_args)

    # Load model checkpoint using strict=False,
    if os.path.exists(cfg.pretrain_ckpt_file):
        logger.info(f'Loading pretrained checkpoint {cfg.pretrain_ckpt_file} ...')
        ckpt_state = torch.load(cfg.pretrain_ckpt_file)
        model.load_state_dict(ckpt_state, strict=False)

    # freeze all layers except output head layer
    for n, p in model.named_parameters():
        if "scalar_head" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported()
    train_dtype = torch.float32
    if cfg.mixed_precision:
        if bf16_ready:
            train_dtype = torch.bfloat16
        else:
            train_dtype = torch.float16
    else:
        logger.warning('Training in float32 mode, make sure you have enough GPU RAM')

    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=train_dtype)

    model = model.to(local_rank)

    # BUG in pytorch 2.0.1, as we found out using torch.autocast will increase GPU RAM usage, and cause CUDA OUT OF MEMORY error
    # when run the training script on a single RTX 3090
    scaler = None
    mp_ctx = nullcontext()

    if cfg.compile_model:
        logger.info('compile model using torch.compile() ...')
        model = torch.compile(model)

    logger.info('Initialize optimizer ...')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        eps=cfg.adamw_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adamw_betas,
        fused=cfg.adamw_fused,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        min_lr=cfg.min_lr,
        max_lr=cfg.max_lr,
        warmup_steps=cfg.warmup_steps,
        max_decay_steps=cfg.max_decay_steps,
    )

    # --------------- Start Training ---------------

    logger.info(f'\nStarting to run {cfg.max_train_iters} training iterations ...')

    torch_profiler = None
    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    tb_writer = None
    memmax = None
    mem_alloc_tracker = None
    inner_pbar = None
    train_stats = val_stats = None

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.model_type))

        if cfg.track_gpu_mem_usage:
            memmax = Memory_Maximizer()
            mem_alloc_tracker = []
            memmax.start()

        inner_pbar = tqdm.tqdm(range(cfg.max_train_iters), colour='blue', desc='Training iterations')

    model.train()
    for iter in range(1, cfg.max_train_iters + 1):
        train_stats = run_single_train_step(
            ctx=mp_ctx,
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

            if cfg.track_gpu_mem_usage:
                memmax.update()
                mem_alloc_tracker.append(format_to_gb(torch.cuda.memory_allocated()))

        # checkpointing
        if cfg.ckpt_interval > 0 and iter % cfg.ckpt_interval == 0 or iter == cfg.max_train_iters:
            # save model state
            checkpoint = model.state_dict()

            torch.save(checkpoint, os.path.join(cfg.ckpt_dir, f'lora_{cfg.model_type}-iter-{iter}.pth'))

        # validation steps
        if cfg.val_iters > 0 and (cfg.val_interval > 0 and iter % cfg.val_interval == 0 or iter == cfg.max_train_iters):
            val_stats = run_validation_steps(ctx=mp_ctx, model=model, rank=rank, world_size=world_size, val_loader=val_loader)

            if rank == 0:
                logger.info(
                    f'Training iteration {iter}: validation loss: {val_stats["loss"]:.4f}, validation accuracy: {val_stats["accuracy"]:.2f}%'
                )

                if tb_writer is not None:
                    tb_writer.add_scalar('val/loss', val_stats['loss'], iter)
                    tb_writer.add_scalar('val/accuracy', val_stats['accuracy'], iter)

    if rank == 0:
        # training is done...show some training stats.
        if cfg.track_gpu_mem_usage:
            memmax.stop()
            logger.info(f'Total memory allocated: {mem_alloc_tracker}')
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
