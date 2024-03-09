# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, Optional, Union, List, Mapping, Text, Any
import logging
import numpy as np
import math
import torch
import torch.distributed as dist
from torch.distributed.fsdp.api import ShardingStrategy


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from instruct_llama.models.model import Transformer


logger = logging.getLogger(__name__)


def make_model_layer_trainable(model: Transformer, trainable_layers: List[str]):
    for n, p in model.named_parameters():
        if trainable_layers and any((train_n in n or train_n == n for train_n in trainable_layers)):
            p.requires_grad = True
        else:
            p.requires_grad = False


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
    model: Transformer,
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

    kwargs = {'lr': lr, 'eps': eps, 'betas': betas}

    if paged_adamw:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW(optim_groups, **kwargs)
    else:
        kwargs['fused'] = fused
        optimizer = torch.optim.AdamW(optim_groups, **kwargs)

    return optimizer


def compute_num_trainable_params(model: Transformer) -> Tuple[int, int]:
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


def split_indices_into_bins(bin_size: int, max_indices: int, min_indices: int = 0, shuffle: bool = False, drop_last: bool = False) -> List[List[int]]:
    """Split indices to small bins."""

    max_indices = int(max_indices)
    min_indices = int(min_indices)

    if max_indices < bin_size:
        raise ValueError(f'Expect max_indices to be greater than bin_size, got {max_indices} and {bin_size}')

    # Split indices into 'bins' with bin_size.
    if shuffle:
        indices = np.random.permutation(max_indices)
    else:
        indices = np.arange(min_indices, max_indices)

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


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    if dim is not None:
        return (values * mask).sum(dim=dim)
    else:
        return (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: Optional[Union[int, Tuple]] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    if dim is not None:
        return (values * mask).sum(dim=dim) / mask.sum(dim=dim)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    assert torch.is_tensor(mask) and mask.dtype == torch.bool
    assert torch.is_tensor(values) and values.shape == mask.shape

    mask_sum = mask.sum()
    if mask_sum == 0 or mask_sum == 1:
        raise ValueError(f'The sum of the mask is {mask_sum}, which can happen when batch size of mask is 1, try increase the batch size')

    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    whitened *= mask.float()
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
    return_norm = local_norm.clone().detach().requires_grad_(False).to(rank) ** 2
    dist.all_reduce(return_norm, op=dist.ReduceOp.SUM)
    if sharding_strategy == ShardingStrategy.NO_SHARD:
        return_norm = return_norm / world_size
    return return_norm**0.5


def optimizer_to(optim: torch.optim.Optimizer, device: str):
    """Move pytorch optimizer to some device

    Code copied from
    https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
