# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, List, Mapping, Text, Any
import logging
import numpy as np
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp.api import ShardingStrategy

import bitsandbytes as bnb

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

logger = logging.getLogger(__name__)


def create_trace_profiler(tb_trace_dir: str) -> torch.profiler.profile:
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
    model: torch.nn.Module,
    lr: float,
    eps: float,
    weight_decay: float,
    betas: Tuple[float],
    fused: bool = False,
    paged_adamw: bool = False,
) -> torch.optim.AdamW:
    """
    Returns the PyTorch AdamW optimizer for the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    # Create empty lists to store parameters for weight decay and no weight decay.
    decay = []
    no_decay = []

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad

        if is_trainable:
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

    if weight_decay > 0:
        num_decay_params = sum(p.numel() for p in decay)
        num_nodecay_params = sum(p.numel() for p in no_decay)
        logger.info(f'Number of decayed parameters: {num_decay_params:,}')
        logger.info(f'Number of non-decayed parameters: {num_nodecay_params:,}')

    # create the pytorch optimizer object
    optim_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    kwargs = {
        'lr': lr,
        'eps': eps,
        'betas': betas,
    }

    if paged_adamw:
        optimizer = bnb.optim.PagedAdamW(optim_groups, **kwargs)
    else:
        kwargs['fused'] = fused
        optimizer = torch.optim.AdamW(optim_groups, **kwargs)

    return optimizer


def compute_num_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    num_trainable_params = 0
    num_frozen_params = 0

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad
        is_quantized = hasattr(params, 'quant_state')

        # quantized layer is not trainable
        if not is_trainable and is_quantized:
            num_params = math.prod(params.quant_state.shape)
        else:
            num_params = params.numel()

        num_trainable_params += num_params if is_trainable else 0
        num_frozen_params += num_params if not is_trainable else 0

    return num_trainable_params, num_frozen_params


def split_indices_into_bins(
    bin_size: int, max_indices: int, min_indices: int = 0, shuffle: bool = False, drop_last: bool = False
) -> List[List[int]]:
    """Split indices to small bins."""

    bin_size = int(bin_size)
    max_indices = int(max_indices)
    min_indices = int(min_indices)

    if max_indices < bin_size:
        raise ValueError(f'Expect max_indices to be greater than bin_size, got {max_indices} and {bin_size}')

    # Split indices into 'bins' with bin_size.
    indices = np.arange(min_indices, max_indices)

    if shuffle:
        np.random.shuffle(indices)

    indices_list = []
    for i in range(0, len(indices), bin_size):
        indices_list.append(indices[i : i + bin_size])  # noqa: E203

    if len(indices_list[-1]) != bin_size:
        if drop_last:
            indices_list = indices_list[:-1]
        else:
            # Make sure the last one has the same 'bin_size'.
            indices_list[-1] = indices[-bin_size:]

    return indices_list


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)

    # avoid division by zero
    mask_sum = torch.where(mask_sum <= 0, 1e-8, mask_sum)

    mean = tensor / mask_sum
    return mean


def masked_sum(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    sum = tensor.sum(dim=dim)
    return sum


def masked_whiten(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8, shift_mean: bool = True
) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)

    if len(tensor.shape) > len(mean.shape) and len(mean.shape) == 1:
        mean = mean.unsqueeze(1)

    mean_centered = tensor - mean

    var = masked_mean(mean_centered**2, mask, dim=dim)
    if len(tensor.shape) > len(var.shape) and len(var.shape) == 1:
        var = var.unsqueeze(1)

    var = torch.where(var == 0, 1.0, var)

    whitened = mean_centered * var.clamp(min=eps).rsqrt()

    if not shift_mean:
        whitened += mean

    return whitened


def get_grad_norm_local(model) -> torch.Tensor:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return total_norm**0.5


def get_grad_norm_fsdp(model, rank, world_size, sharding_strategy=ShardingStrategy.FULL_SHARD) -> torch.Tensor:
    local_norm = get_grad_norm_local(model)
    op = torch.distributed.ReduceOp.SUM
    return_norm = local_norm.clone().detach().requires_grad_(False).to(rank) ** 2
    dist.all_reduce(return_norm, op=op)
    if sharding_strategy == ShardingStrategy.NO_SHARD:
        return_norm = return_norm / world_size
    return return_norm**0.5
