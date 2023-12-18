# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
LLaMA model with LoRALiner layers
"""
import logging
from functools import partial
from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, Iterable, Dict

import numpy as np
import torch
from torch import nn

import instruct_llama.model as llama
from instruct_llama.lora import LoRALinear, LoRALinear4bit, Linear4bit


logger = logging.getLogger(__name__)


@dataclass
class LoraModelArgs(llama.ModelArgs):
    lora_r: float = 0.0
    lora_scaling: float = 1.0
    lora_dropout: float = 0.0

    lora_attn_query: bool = True  # train Attention query layer
    lora_attn_key: bool = False  # train Attention key layer
    lora_attn_value: bool = True  # train Attention value layer
    lora_attn_proj: bool = False  # train Attention output projection layer
    lora_attn_mlp: bool = False  # train Attention MLP block

    quant_4bit: bool = True  # quantize frozen linear layer
    quant_lora_4bit: bool = True  # quantize LoRA linear layer
    quant_4bit_double: bool = True
    quant_4bit_type: str = 'nf4'
    quant_compute_dtype: torch.dtype = torch.bfloat16


def _get_lora_kwargs(params: LoraModelArgs) -> Dict:
    return {
        'r': params.lora_r,
        'lora_scaling': params.lora_scaling,
        'lora_dropout': params.lora_dropout,
    }


def _get_quant_kwargs(params: LoraModelArgs) -> Dict:
    return {
        'compress_statistics': params.quant_4bit_double,
        'quant_type': params.quant_4bit_type,
        'compute_dtype': params.quant_compute_dtype,
    }


def _get_lora_linear_layer(params: LoraModelArgs) -> Union[LoRALinear, LoRALinear4bit]:
    layer_cls = None
    kwargs = _get_lora_kwargs(params)
    if params.quant_lora_4bit:
        kwargs.update(_get_quant_kwargs(params))
        layer_cls = LoRALinear4bit
    else:
        layer_cls = LoRALinear

    return partial(layer_cls, **kwargs)


def _get_linear_layer(params: LoraModelArgs) -> Union[nn.Linear, Linear4bit]:
    layer_cls = None
    kwargs = {}
    if params.quant_4bit:
        kwargs.update(_get_quant_kwargs(params))
        layer_cls = Linear4bit
    else:
        layer_cls = nn.Linear

    return partial(layer_cls, **kwargs)


class Attention(llama.Attention):
    def __init__(self, params: LoraModelArgs) -> None:
        """Attention with training q, v weights using Low Ranking Adaptation for
        parameter-efficient fine-tuning, and keep k, o fixed.

        Args:
            params:
                ``"max_seq_len"``: size of the context of the model,
                ``"vocab_size"``: number of unique tokens,
                ``"n_layers"``: number of transformer blocks (self-attention + MLP),
                ``"n_heads"``: number of heads in multi-head attention mechanism,
                ``"dim"``: size of the embedding: vector representation of each token.
        """
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        self.max_batch_size = params.max_batch_size
        self.max_seq_len = params.max_seq_len
        self.n_heads = params.n_heads
        self.head_dim = params.dim // params.n_heads

        lora_linear_cls = _get_lora_linear_layer(params)
        linear_cls = _get_linear_layer(params)

        query_layer_cls = lora_linear_cls if params.lora_attn_query else linear_cls
        key_layer_cls = lora_linear_cls if params.lora_attn_key else linear_cls
        value_layer_cls = lora_linear_cls if params.lora_attn_value else linear_cls
        proj_layer_cls = lora_linear_cls if params.lora_attn_proj else linear_cls

        self.wq = query_layer_cls(
            params.dim,
            params.n_heads * self.head_dim,
            bias=False,
        )

        self.wk = key_layer_cls(
            params.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = value_layer_cls(
            params.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = proj_layer_cls(
            params.n_heads * self.head_dim,
            params.dim,
            bias=False,
        )

        self.use_cache = params.use_cache

        self.cache_k = None
        self.cache_v = None

        # regularization
        self.attn_dropout = nn.Dropout(params.attn_dropout) if params.attn_dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(params.resid_dropout) if params.resid_dropout > 0 else nn.Identity()


class FeedForward(llama.FeedForward):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        resid_dropout: Optional[float],
        params: LoraModelArgs,
    ):
        nn.Module.__init__(self)
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if params.lora_attn_mlp:
            layer_cls = _get_lora_linear_layer(params)
        else:
            layer_cls = _get_linear_layer(params)

        self.w1 = layer_cls(dim, hidden_dim, bias=False)
        self.w2 = layer_cls(hidden_dim, dim, bias=False)
        self.w3 = layer_cls(dim, hidden_dim, bias=False)

        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity()


class TransformerBlock(llama.TransformerBlock):
    def __init__(self, layer_id: int, params: LoraModelArgs):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.n_heads = params.n_heads
        self.dim = params.dim
        self.head_dim = params.dim // params.n_heads

        self.attention = Attention(params)
        self.feed_forward = FeedForward(
            dim=params.dim,
            hidden_dim=4 * params.dim,
            multiple_of=params.multiple_of,
            ffn_dim_multiplier=params.ffn_dim_multiplier,
            resid_dropout=params.resid_dropout,
            params=params,
        )
        self.attention_norm = llama.RMSNorm(params.dim, eps=params.norm_eps)
        self.ffn_norm = llama.RMSNorm(params.dim, eps=params.norm_eps)


class Transformer(llama.Transformer):
    def __init__(self, params: LoraModelArgs):
        nn.Module.__init__(self)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.token_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.embeddings_dropout = nn.Dropout(params.embed_dropout) if params.embed_dropout > 0 else nn.Identity()

        self.layers: Iterable[TransformerBlock] = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.post_norm = llama.RMSNorm(params.dim, eps=params.norm_eps)

        # do not apply LoRA or quantize to the lm_head or scalar_head layer
        if self.params.head_type == 'lm_head':
            logger.info('Creating LLaMA-2 model with LM head ...')
            self.lm_head = nn.Linear(params.dim, params.vocab_size, bias=False)
        elif self.params.head_type == 'scalar_head':
            logger.info('Creating LLaMA-2 model with scalar head ...')
            self.scalar_head = nn.Linear(params.dim, 1, bias=True)

        self.freqs_cis = llama.precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
