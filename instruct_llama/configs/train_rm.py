from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class config:
    """Trains Reward Model (RM) using LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    head_type: str = 'scalar_head'  # reward model requires scalar head

    pretrain_ckpt_file: str = './checkpoints/7b-finetune/iter-1000-merged.pth'  # load fine-tuned checkpoint
    tokenizer_file: str = './checkpoints/llama-2/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = ('./datasets/stackexchange_dataset/train.pkl',)
    val_datasources: Tuple[str] = ('./datasets/stackexchange_dataset/validation.pkl',)
    dataloader_workers: int = 1

    max_seq_len: int = 2048  # use smaller sequence length to save GPU RAM

    # training and validation loops
    # 34000 training samples / 32 = 1000 iters
    max_train_iters: int = 3000
    # we always use micro batch size of 1, with average completions per sample of 4, this is roughly 128 batch size when doing update
    gradient_accum_steps: int = 32
    val_interval: int = 500
    val_iters: int = 400  # large size since micro_batch_size is very small
    log_interval: int = 100  # log training metrics (loss, accuracy)
    ckpt_interval: int = 1000  # save model checkpoints every N training iterations

    # learning rate scheduler
    init_lr: float = 1e-6  # use a much smaller initial learning rate
    max_lr: float = 4e-5  # max learning rate when warm up
    min_lr: float = 1e-5  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 3000

    # AdamW optimizer
    use_bnb_8bit: bool = False  # use bitsandbytes 8bit optimizer
    weight_decay: float = 0.0
    adamw_betas: Tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_fused: bool = True  # only applicable if not using bitsandbytes 8bit optimizer

    grad_clip: float = 1.0

    # dropout regularization, note using drop out will slows down the training
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # training speed improvement
    mixed_precision: bool = True  # try bfloat16 first, but will use float16 if GPU not support
    compile_model: bool = False

    # others
    seed: int = 127
    log_dir: str = './logs/train_rm'  # save logs and traces
    ckpt_dir: str = './checkpoints/train_rm'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
    track_gpu_mem_usage: bool = False  # track GPU memory allocation statistics
