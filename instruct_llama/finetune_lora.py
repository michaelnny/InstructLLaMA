import os
import itertools
import functools
from typing import Tuple
import tqdm
import random
import numpy as np
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.model import Transformer, ModelArgs
from instruct_llama.tokenizer import Tokenizer
from instruct_llama.utils import FineTuneDataset
from instruct_llama.lora import lora, lora_state_dict, mark_only_lora_as_trainable

from instruct_llama.configs.finetune_lora import config as cfg


from instruct_llama.utils import (
    CosineDecayWithWarmupLRScheduler,
    Memory_Maximizer,
    format_to_gb,
    create_logger,
)


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f"clearing cache for rank {rank}")
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
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[int, float]:
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
            p_name.endswith("bias")
            or p_name.endswith("ln_1.weight")
            or p_name.endswith("ln_2.weight")
            or p_name.endswith("post_ln.weight")
            or p_name.endswith("position_embed.weight")
            or p_name.endswith("token_embed.weight")
        ):
            no_decay.append(params)
        else:
            decay.append(params)

    num_decay_params = sum(p.numel() for p in decay)
    num_nodecay_params = sum(p.numel() for p in no_decay)
    total_num_params = sum(p.numel() for p in params_dict.values())
    assert num_decay_params + num_nodecay_params == total_num_params

    print(f"--> num decayed parameter tensors: {len(decay)}, with {num_decay_params:,} parameters")
    print(f"--> num non-decayed parameter tensors: {len(no_decay)}, with {num_nodecay_params:,} parameters")

    # create the pytorch optimizer object
    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
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

    local_rank = int(os.environ["LOCAL_RANK"])

    if return_stats:
        fsdp_metrics = torch.zeros(5).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    for x, y, loss_mask in itertools.islice(train_loader, cfg.gradient_accum_steps):
        x, y, loss_mask = (
            x.to(local_rank, non_blocking=True),
            y.to(local_rank, non_blocking=True),
            loss_mask.to(local_rank, non_blocking=True),
        )

        with ctx:  # mixed precision
            output = model(x)

        loss = compute_finetune_loss(output, y, loss_mask)
        # scale the loss to account for gradient accumulation
        scaled_loss = loss / cfg.gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if return_stats:
            num_acc, num_samples = compute_metrics(output, y, loss_mask)
            fsdp_metrics[0] += loss.item()  # sum up batch loss
            fsdp_metrics[1] += np.exp(loss.item())  # sum up perplexity
            fsdp_metrics[2] += 1  # increase number of batches
            fsdp_metrics[3] += num_acc  # sum up number of accurate prediction tokens
            fsdp_metrics[4] += num_samples  # sum up number of tokens

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
        train_loss = fsdp_metrics[0] / fsdp_metrics[2]
        train_perplexity = fsdp_metrics[1] / fsdp_metrics[2]
        train_accuracy = 100 * fsdp_metrics[3] / fsdp_metrics[4]

        lr = optimizer.param_groups[0]["lr"]
        return {
            "loss": train_loss.item(),
            "accuracy": train_accuracy.item(),
            "perplexity": train_perplexity.item(),
            "learning_rate": lr,
        }
    else:
        return None


def run_validation_steps(ctx, model, rank, world_size, val_loader):
    """Run M evaluation iterations"""
    model.eval()  # set model in evaluation mode

    local_rank = int(os.environ["LOCAL_RANK"])

    fsdp_metrics = torch.zeros(5).to(local_rank)

    inner_pbar = tqdm.tqdm(range(cfg.val_iters), colour="green", desc="Evaluation iterations")

    with torch.no_grad():
        for x, y, loss_mask in itertools.islice(val_loader, cfg.val_iters):
            x, y, loss_mask = (
                x.to(local_rank, non_blocking=True),
                y.to(local_rank, non_blocking=True),
                loss_mask.to(local_rank, non_blocking=True),
            )

            with ctx:
                output = model(x)

            loss = compute_finetune_loss(output, y, loss_mask)

            num_acc, num_samples = compute_metrics(output, y, loss_mask)
            fsdp_metrics[0] += loss.item()  # sum up batch loss
            fsdp_metrics[1] += np.exp(loss.item())  # sum up perplexity
            fsdp_metrics[2] += 1  # increase number of batches
            fsdp_metrics[3] += num_acc  # sum up number of accurate prediction tokens
            fsdp_metrics[4] += num_samples  # sum up number of tokens

            if inner_pbar is not None:
                inner_pbar.update(1)

    val_loss = fsdp_metrics[0] / fsdp_metrics[2]
    val_perplexity = fsdp_metrics[1] / fsdp_metrics[2]
    val_accuracy = 100 * fsdp_metrics[3] / fsdp_metrics[4]

    inner_pbar.close()

    model.train()  # set model in training mode after evaluation runs

    return {"loss": val_loss.item(), "accuracy": val_accuracy.item(), 'perplexity': val_perplexity.item()}


def custom_collate_fn(batch, pad_id, max_seq_len):
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    which is much faster than pad the sequence to some global max sequence length.

    In addition, it will compute the attention mask and loss mask for the batch.
    """

    batch_size = len(batch)

    batch_seq_lengths = [len(item[0]) + len(item[1]) for item in batch]

    max_batch_seq_length = max(batch_seq_lengths)

    assert max_batch_seq_length <= max_seq_len

    # concatenate prompt, completion together
    batch_sequences = torch.full((batch_size, max_batch_seq_length), pad_id, dtype=torch.long)

    # where -1s are prompt tokens, 1s are completion tokens, and 0s are padding tokens
    loss_mask = torch.full((batch_size, max_batch_seq_length), 0, dtype=torch.long)

    for i, (prompt, completion) in enumerate(batch):
        # need prompt, completion lengths to compute loss mask
        prompt_len, completion_len = len(prompt), len(completion)

        # enforce check sequence length, since trunk sequence is not the ideal solution since we might lost some very important context
        seq_len = prompt_len + completion_len
        assert seq_len <= max_batch_seq_length

        seq = torch.concat((prompt, completion), dim=0)

        # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
        batch_sequences[i, :seq_len] = seq
        loss_mask[i, :prompt_len] = -1  # prompt tokens
        loss_mask[i, prompt_len : prompt_len + completion_len] = 1  # completion tokens

    x = batch_sequences[:, :-1]  # [batch_size, max_batch_seq_length - 1]
    y = batch_sequences[:, 1:]  # [batch_size, max_batch_seq_length - 1]

    # shift to right to align with y
    loss_mask = loss_mask[:, 1:]

    # # create attention mask
    # # BUG in SDPA module when use -inf or bool mask will cause NaNs
    # attn_mask = torch.full((batch_size, 1, max_batch_seq_length - 1, max_batch_seq_length - 1), float(-1e10))

    # attn_mask = torch.triu(attn_mask, diagonal=1)

    return x, y, loss_mask


def main():
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint "{cfg.pretrain_ckpt_file}", aborting...')

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup()

    logger = create_logger()

    # --------------- Load datasets ---------------

    logger.info("\nLoading datasets ...")

    tokenizer = Tokenizer(cfg.tokenizer_file)

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=0,  # tokenizer.pad_id,
        max_seq_len=cfg.max_seq_len,
    )

    train_dataset = FineTuneDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)

    cuda_kwargs = {
        "collate_fn": _collate_fn,
        "num_workers": cfg.dataloader_workers,
        "batch_size": cfg.micro_batch_size,
        "pin_memory": True,
        "shuffle": False,
    }

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)

    train_kwargs = {"batch_size": cfg.micro_batch_size, "sampler": train_sampler}

    train_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs)

    logger.info(f"--> Train dataset metadata:\n{train_dataset.get_metadata()}")

    # create validation dataset
    val_loader = None
    if cfg.val_interval > 0:
        val_dataset = FineTuneDataset(data_sources=cfg.val_datasources, max_seq_len=cfg.max_seq_len)

        val_kwargs = {"batch_size": cfg.micro_batch_size}
        val_kwargs.update(cuda_kwargs)

        val_loader = DataLoader(val_dataset, **val_kwargs)

        logger.info(f"--> Evaluation dataset metadata:\n{val_dataset.get_metadata()}")

    # --------------- Setup model and optimizer ---------------

    logger.info("\nInitialize model and optimizer ...")

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    with lora(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, enabled=True):
        model_args = ModelArgs.from_model_type(cfg.model_type)
        model_args.vocab_size = tokenizer.vocab_size
        model_args.head_type = "lm_head"
        model_args.max_seq_len = cfg.max_seq_len
        model_args.embed_dropout = cfg.embed_dropout
        model_args.attn_dropout = cfg.attn_dropout
        model_args.resid_dropout = cfg.resid_dropout

        model = Transformer(model_args)

        # Load model checkpoint before passing into FSDP
        if os.path.exists(cfg.pretrain_ckpt_file):
            logger.info(f"--> load pretrained checkpoint {cfg.pretrain_ckpt_file}")
            ckpt_state = torch.load(cfg.pretrain_ckpt_file)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            model.load_state_dict(ckpt_state, strict=False)

    mark_only_lora_as_trainable(model, bias=cfg.train_bias, head=cfg.train_head)

    # train the model in half percision
    model = model.half()
    model = model.to(local_rank)

    scaler = torch.cuda.amp.GradScaler()

    bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported()

    mp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16_ready else torch.float16)
        if cfg.mixed_precision
        else nullcontext()
    )

    if cfg.compile_model:
        logger.info(f"--> compile model using torch.compile() ...")
        model = torch.compile(model)

    logger.info("\nInitialize optimizer ...")

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

    logger.info(f"\nStarting to run {cfg.max_train_iters} training iterations ...")

    torch_profiler = None
    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, "profile_traces"))

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

        inner_pbar = tqdm.tqdm(range(cfg.max_train_iters), colour="blue", desc="Training iterations")

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
                f'Training iteration {iter}: train loss: {train_stats["loss"]:.4f}, '
                f'train accuracy: {train_stats["accuracy"]:.2f}%, train perplexity: {train_stats["perplexity"]:.2f}, learning rate: {train_stats["learning_rate"]:.10f}'
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", train_stats["loss"], iter)
                tb_writer.add_scalar("train/accuracy", train_stats["accuracy"], iter)
                tb_writer.add_scalar("train/perplexity", train_stats["perplexity"], iter)
                tb_writer.add_scalar("train/learning_rate", train_stats["learning_rate"], iter)

            if cfg.track_gpu_mem_usage:
                memmax.update()
                mem_alloc_tracker.append(format_to_gb(torch.cuda.memory_allocated()))

        # checkpointing
        if cfg.ckpt_interval > 0 and iter % cfg.ckpt_interval == 0 or iter == cfg.max_train_iters:
            # save model state
            checkpoint = lora_state_dict(model, bias=cfg.train_bias, head=cfg.train_head)

            torch.save(checkpoint, os.path.join(cfg.ckpt_dir, f"lora_{cfg.model_type}-iter-{iter}.pt"))

        # validation steps
        if cfg.val_iters > 0 and (cfg.val_interval > 0 and iter % cfg.val_interval == 0 or iter == cfg.max_train_iters):
            val_stats = run_validation_steps(ctx=mp_ctx, model=model, rank=rank, world_size=world_size, val_loader=val_loader)

            if rank == 0:
                logger.info(
                    f'Training iteration {iter}: evaluation loss: {val_stats["loss"]:.4f}, '
                    f'evaluation accuracy: {val_stats["accuracy"]:.2f}%, evaluation perplexity: {val_stats["perplexity"]:.2f}'
                )

                if tb_writer is not None:
                    tb_writer.add_scalar("eval/loss", val_stats["loss"], iter)
                    tb_writer.add_scalar("eval/accuracy", val_stats["accuracy"], iter)
                    tb_writer.add_scalar("eval/perplexity", val_stats["perplexity"], iter)

    if rank == 0:
        # training is done...show some training stats.
        if cfg.track_gpu_mem_usage:
            memmax.stop()
            logger.info(f"Total memory allocated: {mem_alloc_tracker}")
            logger.info(f"CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}")

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # torch.set_float32_matmul_precision("high")

    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # torch.backends.cuda.enable_math_sdp(True)

    main()