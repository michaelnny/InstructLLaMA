from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class config:
    """Trains Reward Model (RM) using LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    head_type: str = 'scalar_head'
    max_seq_len: int = 450  # RM model is special because we need to maintain two graphs when computing loss, so the sequence length is much shorter

    rm_ckpt_file: str = './checkpoints/7b-sft/iter-2000-merged.pth'  # load fine-tuned checkpoint
    tokenizer_file: str = './meta_checkpoints/llama-2/tokenizer.model'  # load tokenizer model

    random_head_weights: bool = True

    # datasets
    train_datasources: Tuple[str] = (
        # './datasets/stackexchange_dataset/train.pkl', # this demands more GPU RAM because we have more than 2 responses per each sample
        './datasets/hh-rlhf/train.pkl',
    )
    val_datasources: Tuple[str] = (
        # './datasets/stackexchange_dataset/validation.pkl',
        './datasets/hh-rlhf/validation.pkl',
    )
    dataloader_workers: int = 1

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    # we always use micro batch size of 1 (sample) during training and evaluation
    gradient_accum_steps: int = 64
    # 120000 training samples / 64 = 2000 iters per epoch
    max_train_iters: int = 2000 * 5
    val_interval: int = 200  # this also decides how often we create model checkpoints
    val_iters: int = 512
    log_interval: int = 10  # log training metrics (loss, accuracy)

    # whether normalize reward before compute loss during training and validation
    normalize_reward: bool = False

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    train_bias: str = 'none'  # none, lora_only, all
    train_head: bool = True  # train scalar_head is required since the output layer is initialized randomly

    # learning rate scheduler, if also train head, should use a smaller learning rate
    init_lr: float = 1e-6  # use a much smaller initial learning rate
    max_lr: float = 1e-5  # max learning rate when warm up
    min_lr: float = 5e-6  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 2000 * 2

    # AdamW optimizer
    use_bnb_8bit: bool = False  # use bitsandbytes 8bit optimizer
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_fused: bool = True  # only applicable if not using bitsandbytes 8bit optimizer
    grad_clip: float = 0.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # FSDP module
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    # when use fsdp_activation_checkpointing, will see 30-50% training slowdown, but can free up ~30% GPU RAM thus we can use larger batch size
    fsdp_activation_checkpointing: bool = False
    forward_prefetch: bool = False  # not working with LoRA
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE  # BACKWARD_PRE, BACKWARD_POST
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  # FULL_SHARD, HYBRID_SHARD, SHARD_GRAD_OP, NO_SHARD
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT  # alternatively can use SHARDED_STATE_DICT to avoid OOMs
    cpu_offload: bool = False
    limit_all_gathers: bool = True
    use_orig_params: bool = True
    compile_model: bool = False  # BUG in torch 2.0.1 when working with FSDP, not support python 3.11

    # others
    seed: int = 143
    log_dir: str = './logs/train_rm_fsdp_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/train_rm_fsdp_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
