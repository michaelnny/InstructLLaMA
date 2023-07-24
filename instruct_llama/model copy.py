"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self


MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class ModelArgs:
    max_seq_len: int = 2048  # maximum model context size
    max_batch_size: int = 4
    vocab_size: int = 32000  # make sure it's multiple of 64 for better speed
    num_layers: int = 32
    num_attn_heads: int = 32
    hidden_size: int = 4096
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2

    norm_eps: float = 1e-5
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


# llama 2 models
llama_configs = {
    "toy": dict(
        num_layers=4, num_attn_heads=8, hidden_size=512
    ),  # toy model for testing code only
    "7B": dict(num_layers=32, num_attn_heads=32, hidden_size=4096),
    "13B": dict(num_layers=40, num_attn_heads=40, hidden_size=5120),
    "70B": dict(num_layers=80, num_attn_heads=64, hidden_size=8192),
}


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        assert config.hidden_size % config.num_attn_heads == 0

        self.num_attn_heads = config.num_attn_heads
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_len
        self.head_dim = self.hidden_size // config.num_attn_heads

        # key, query, value projections for all heads, but in a batch
        self.wq = nn.Linear(
            self.hidden_size, self.num_attn_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            self.hidden_size, self.num_attn_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            self.hidden_size, self.num_attn_heads * self.head_dim, bias=False
        )

        # output projection
        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # regularization
        self.resid_drop = nn.Dropout(config.resid_dropout)
        self.attn_dropout = config.attn_dropout

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        # get batch size, sequence length
        B, L, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        k = k.view(B, L, self.num_attn_heads, self.head_dim)
        q = q.view(B, L, self.num_attn_heads, self.head_dim)
        v = v.view(B, L, self.num_attn_heads, self.head_dim)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, num_attn_heads, L, head_dim)
        q = q.transpose(1, 2)  # (B, num_attn_heads, L, head_dim)
        v = v.transpose(1, 2)  # (B, num_attn_heads, L, head_dim)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True if mask is None else False,
        )
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, L, C)

        # output projection
        y = self.wo(y)
        y = self.resid_drop(y)

        return y, kv_cache


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        hidden_dim = 4 * config.hidden_size
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.resid_drop = nn.Dropout(config.resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        x = self.resid_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kv_cache = self.attention(
            self.attention_norm(x), rope, mask, max_seq_length, input_pos, kv_cache
        )
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs, head_type: str = "lm") -> None:
        super().__init__()
        assert head_type in {"none", "lm", "scalar"}
        self.config = config
        self.head_type = head_type

        self.tokens_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.embed_drop = nn.Dropout(self.config.embed_dropout)

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )

        # norm layer applied after transformer layers
        self.post_norm = RMSNorm(config.hidden_size)

        if head_type == "lm":  # language modeling
            print("Creating LLaMA model with language modeling head...")
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        elif head_type == "scalar":  # reward or value modeling
            print("Creating LLaMA model with scalar head...")
            self.scalar_head = nn.Linear(config.hidden_size, 1, bias=True)
        else:
            print("Creating LLaMA model without output head...")

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.num_layers),
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.num_layers),
            )

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        max_seq_len = self.config.max_seq_len
        if max_seq_length is None or max_seq_length < 1:
            max_seq_length = max_seq_len
        assert (
            T <= max_seq_length
        ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert (
            max_seq_length <= max_seq_len
        ), f"Cannot attend to {max_seq_length}, block size is only {max_seq_len}"
        assert (
            T <= max_seq_len
        ), f"Cannot forward sequence of length {T}, block size is only {max_seq_len}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if input_pos is not None:
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.tokens_embedding(idx)  # token embeddings of shape (b, t, hidden_size)
        x = self.embed_drop(x)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.layers:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.hidden_size // self.config.num_attn_heads
                cache_shape = (B, self.config.num_attn_heads, max_seq_length, head_size)
                self.kv_caches = [
                    (
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                    )
                    for _ in range(self.config.num_layers)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x, rope, mask, max_seq_length, input_pos, self.kv_caches[i]
                )

        x = self.post_norm(x)

        if self.head_type == "lm":
            logits = self.lm_head(x)  # (b, t, vocab_size)
            return logits
        elif self.head_type == "scalar":
            scalar = self.scalar_head(x)  # (b, t, 1)
            return scalar
        else:
            return x  # hidden state

    @classmethod
    def from_name(cls, name: str, head_type: str = "lm") -> Self:
        return cls(ModelArgs.from_name(name), head_type)

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.max_seq_len,
            n_elem=self.config.hidden_size // self.config.num_attn_heads,
            dtype=idx.dtype,
            device=idx.device,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> MaskCache:
        ones = torch.ones(
            (self.config.max_seq_len, self.config.max_seq_len),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
