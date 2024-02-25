# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from instruct_llama.utils.custom_dataset import DataSource


@dataclass
class config:
    """Pre-training"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '1B'  # 7B, 13B, 70B
    head_type: str = 'lm_head'
    max_seq_len: int = 512

    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[DataSource] = (
        DataSource(
            name='red_pajama_mini',
            weights=1.0,
            data_file='./datasets/red_pajama_mini/train.npy',
            metadata_file='./datasets/red_pajama_mini/train_meta.json',
        ),
    )
    val_datasources: Tuple[DataSource] = (
        DataSource(
            name='red_pajama_mini',
            weights=1.0,
            data_file='./datasets/red_pajama_mini/validation.npy',
            metadata_file='./datasets/red_pajama_mini/validation_meta.json',
        ),
    )
    dataloader_workers: int = 1

    # training and validation loops
    num_epochs: int = 5
    # accumulate gradients so for each iteration, the actual batch size is = train_batch_size x gradient_accum_steps
    train_batch_size: int = 4
    gradient_accum_steps: int = 8
    loss_scale: float = 1.0  # scale loss to account for gradient accumulation, we don't want to use a very small scale
    val_interval: int = 1000
    val_batch_size: int = 30
    val_steps: int = 20
    log_interval: int = 5  # log training metrics (loss, accuracy)
    ckpt_interval: int = 1000  # save model checkpoints every N Training steps

    # learning rate
    init_lr: float = 5e-6  # initial learning rate
    max_lr: float = 5e-5  # max learning rate after warm up
    min_lr: float = 5e-6  # min learning rate after decay
    warmup_ratio: float = 0.1

    # AdamW optimizer
    weight_decay: float = 0.01
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = True
    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0

    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.

    # FSDP
    # when use fsdp_activation_checkpointing, will see 30-50% training slowdown, but can free up ~30% GPU RAM thus we can use larger batch size
    fsdp_activation_checkpointing: bool = False
    # set this to true will cause RuntimeError: Cannot writeback when the gradient shape changes in pytorch 2.1
    forward_prefetch: bool = False
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE  # BACKWARD_PRE, BACKWARD_POST
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  # FULL_SHARD, HYBRID_SHARD, SHARD_GRAD_OP, NO_SHARD
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT  # alternatively can use SHARDED_STATE_DICT to avoid OOMs
    cpu_offload: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True  # required if we want to apply weights decay to certain layers

    # others
    seed: int = 113
    log_dir: str = './logs/pretrain'  # save logs and traces
    ckpt_dir: str = './checkpoints/pretrain'
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
