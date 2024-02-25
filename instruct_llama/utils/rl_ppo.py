# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import List
import os
import numpy as np
import torch


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model_lora import Transformer, LoraModelArgs
from instruct_llama.configs.rlhf_lora import config as RunConfig


def convert_model_to_dtype(model: Transformer, compute_dtype: torch.dtype):
    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=compute_dtype)


def build_model(
    model_cfg: RunConfig,
    vocab_size: int,
    ckpt_file: str,
    device: str,
    compute_dtype,
    model_type: str,
    frozen: bool = True,
    head_type: str = 'lm_head',
    strict: bool = True,
) -> Transformer:
    assert vocab_size > 0

    model_args = LoraModelArgs.from_model_type(
        model_type=model_type,
        vocab_size=vocab_size,
        # LoRA configurations, note if it's for inference then no need to user lora
        lora_r=0 if frozen else model_cfg.lora_r,
        lora_scaling=0 if frozen else model_cfg.lora_scaling,
        lora_dropout=0 if frozen else model_cfg.lora_dropout,
        # LoRA trainable layers, not need to apply LoRA if not trainable
        lora_attn_query=False if frozen else model_cfg.lora_attn_query,
        lora_attn_key=False if frozen else model_cfg.lora_attn_key,
        lora_attn_value=False if frozen else model_cfg.lora_attn_value,
        lora_attn_proj=False if frozen else model_cfg.lora_attn_proj,
        lora_attn_mlp=False if frozen else model_cfg.lora_attn_mlp,
        # Quantization configurations
        quant_4bit=True if frozen else model_cfg.quant_4bit,  # always quantize frozen model to save GPU RAM
        quant_lora_4bit=False if frozen else model_cfg.quant_lora_4bit,
        quant_4bit_double=True if frozen else model_cfg.quant_4bit_double,
        quant_4bit_type=model_cfg.quant_4bit_type,
        quant_compute_dtype=compute_dtype,
        # Regular configurations
        head_type=head_type,
        use_cache=False,
        max_seq_len=model_cfg.max_seq_len,
        max_batch_size=(
            model_cfg.selfplay_batch_size if (head_type == 'dual_head' or head_type == 'lm_head') and not frozen else 1
        ),
        embed_dropout=0.0 if frozen else model_cfg.embed_dropout,
        attn_dropout=0.0 if frozen else model_cfg.attn_dropout,
        gradient_checkpointing=False if frozen else model_cfg.gradient_checkpointing,
    )

    model = Transformer(model_args)

    if os.path.exists(ckpt_file):
        print(f'Loading model checkpoint {ckpt_file!r} ...')
        model_state = torch.load(ckpt_file)
        model.load_state_dict(model_state, strict=strict)
        del model_state  # free up CPU RAM

    if device == 'cpu':
        convert_model_to_dtype(model, torch.float32)
    else:
        convert_model_to_dtype(model, compute_dtype)

    if frozen:
        for p in model.parameters():
            p.requires_grad = False

    return model.to(device)


def clip_reward(x: torch.Tensor, max_abs_reward: float) -> torch.Tensor:
    if max_abs_reward > 0:
        return torch.clamp(x, min=-max_abs_reward, max=max_abs_reward)
    else:
        return x


def truncated_generalized_advantage_estimation(
    r_t: torch.Tensor,
    value_t: torch.Tensor,
    value_tp1: torch.Tensor,
    discount_tp1: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ + γₜ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Args:
      r_t: Sequence of rewards at times [0, k]
      value_t: Sequence of values under π at times [0, k]
      value_tp1: Sequence of values under π at times [1, k+1]
      discount_tp1: Sequence of discounts at times [1, k+1]
      lambda_: a scalar

    Returns:
      Multistep truncated generalized advantage estimation at times [0, k].
    """

    assert len(r_t.shape) == 1
    assert len(value_t.shape) == 1
    assert len(value_tp1.shape) == 1
    assert len(discount_tp1.shape) == 1

    lambda_ = torch.ones_like(discount_tp1) * lambda_  # If scalar, make into vector.

    delta_t = r_t + discount_tp1 * value_tp1 - value_t

    advantage_t = torch.zeros_like(delta_t, dtype=torch.float32)

    gae_t = 0
    for i in reversed(range(len(delta_t))):
        gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
        advantage_t[i] = gae_t

    return advantage_t


def find_begin_of_pattern(input_list: List[int], pattern: List[int] = [518, 25580, 29962]) -> int:
    """Find the beginning index of the some special token patterns from the given list"""
    assert len(pattern) > 1
    assert len(input_list) > len(pattern)

    pattern_length = len(pattern)
    lst_length = len(input_list)
    for i in range(lst_length - pattern_length + 1):
        if input_list[i : i + pattern_length] == pattern:
            return i

    return -1  # Return -1 if pattern is not found


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float = 0.02, adaptive: bool = False, target: float = 1.0, horizon: int = 10000) -> None:
        assert init_kl_coef >= 0
        assert target > 0
        assert horizon >= 100

        self.kl_coef = init_kl_coef
        self.adaptive = adaptive
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        if not self.adaptive:
            return

        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult

    @property
    def value(self) -> float:
        return self.kl_coef
