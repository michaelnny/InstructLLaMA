from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class config:
    """fine-tuning configurations, where we use smaller learning rates, and less training steps"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = "7B"  # 7B, 13B, 70B

    pretrain_ckpt_file = "./checkpoints/llama-2/llama-2-7b/consolidated.pth"  # load pretrained checkpoint
    tokenizer_file = "./checkpoints/llama-2/tokenizer.model"  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = (
        "./datasets/dolly/train.pkl",
        "./datasets/alpaca/train.pkl",
        "./datasets/deepmind_mathematics/train.pkl",
    )
    val_datasources: Tuple[str] = (
        "./datasets/dolly/validation.pkl",
        "./datasets/alpaca/validation.pkl",
        "./datasets/deepmind_mathematics/validation.pkl",
    )
    dataloader_workers: int = 1

    max_seq_len: int = 128  # use smaller sequence length to save GPU RAM

    # training and validation loops
    # training samples * epochs / batch size, 600000 training samples, with batch size of 120, 4000 iters = one epoch
    max_train_iters: int = 20000
    # accumulate gradients so for each iteration, the actual batch size is = micro_batch_size x gradient_accum_steps
    micro_batch_size: int = 1
    gradient_accum_steps: int = 5
    val_interval: int = 50
    val_iters: int = 50  # large size since micro_batch_size is very small
    log_interval: int = 20  # log training metrics (loss, accuracy)
    ckpt_interval: int = 100  # save model and optionally optimizer checkpoints every N training iterations

    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    train_bias: str = "none"  # none, lora_only, all
    train_head: str = "none"  # none, lm_head, scalar_head

    # learning rate scheduler
    init_lr: float = 1e-8  # initial learning rate
    max_lr: float = 2e-6  # max learning rate when warm up, 0.02 x pre-training learning rate
    min_lr: float = 2e-7  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 20000

    # prompt is less important than completion
    prompt_loss_weight: float = 0.01
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    weight_decay: float = 0.0
    adamw_betas: Tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_fused: bool = True

    use_bnb_8bit: bool = True  # use bitsandbytes 8bit optimizer

    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.1
    attn_dropout: float = 0.2
    resid_dropout: float = 0.2

    # training speed improvement
    mixed_precision: bool = True  # try BF16, but will use FP16 if GPU not support
    compile_model: bool = False  # not support python 3.11 yet

    # others
    seed: int = 127
    log_dir: str = "./logs/finetune_lora"  # save logs and traces
    ckpt_dir: str = "./checkpoints/finetune_lora"
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
    track_gpu_mem_usage: bool = True  # track GPU memory allocation statistics
