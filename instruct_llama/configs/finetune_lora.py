from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class config:
    """fine-tuning configurations, where we use smaller learning rates, and less training steps"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    head_type: str = 'lm_head'

    pretrain_ckpt_file: str = './checkpoints/llama-2/llama-2-7b/consolidated.pth'  # load pretrained checkpoint
    tokenizer_file: str = './checkpoints/llama-2/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = (
        './datasets/dolly/train.pkl',
        './datasets/alpaca/train.pkl',
        './datasets/squad/train.pkl',
        './datasets/commonsense_dialogues/train.pkl',
    )
    val_datasources: Tuple[str] = (
        './datasets/dolly/validation.pkl',
        './datasets/alpaca/validation.pkl',
        './datasets/squad/validation.pkl',
        './datasets/commonsense_dialogues/validation.pkl',
    )
    dataloader_workers: int = 1

    max_seq_len: int = 450  # use smaller sequence length to save GPU RAM

    # training and validation loops
    # training samples * epochs / batch size, 70000 training samples, with batch size of 128, 500 iters = one epoch
    max_train_iters: int = 1000
    # accumulate gradients so for each iteration, the actual batch size is = micro_batch_size x gradient_accum_steps
    micro_batch_size: int = 2
    gradient_accum_steps: int = 64
    val_interval: int = 500
    val_iters: int = 500  # large size since micro_batch_size is very small
    log_interval: int = 100  # log training metrics (loss, accuracy)
    ckpt_interval: int = 500  # save model checkpoints every N training iterations

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    train_bias: str = 'none'  # none, lora_only, all
    train_head: str = 'none'  # none, lm_head, scalar_head, the performance is not great when also train head

    # learning rate scheduler
    init_lr: float = 2e-7  # use a much smaller initial learning rate
    max_lr: float = 2e-5  # max learning rate when warm up
    min_lr: float = 2e-6  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 1000

    # prompt is less important than completion
    prompt_loss_weight: float = 0.01
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    use_bnb_8bit: bool = True  # use bitsandbytes 8bit optimizer
    weight_decay: float = 0.0
    adamw_betas: Tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_fused: bool = True  # only applicable if not using bitsandbytes 8bit optimizer

    grad_clip: float = 1.0

    # dropout regularization, note using drop out will slows down the training
    embed_dropout: float = 0.2
    attn_dropout: float = 0.2
    resid_dropout: float = 0.2

    # training speed improvement
    mixed_precision: bool = True  # try bfloat16 first, but will use float16 if GPU not support
    compile_model: bool = False

    # others
    seed: int = 127
    log_dir: str = './logs/finetune_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/finetune_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
    track_gpu_mem_usage: bool = False  # track GPU memory allocation statistics
