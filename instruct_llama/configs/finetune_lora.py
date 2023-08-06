from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class config:
    """Supervised fine-tuning uisng LoRA"""

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

    max_seq_len: int = 512  # use smaller sequence length to save GPU RAM

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we will soon found out if CUDA out of memory error occurs, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    # training samples * epochs / batch size, 70000 training samples, with batch size of 128, 500 iters = one epoch
    max_train_iters: int = 2000
    # accumulate gradients so for each iteration, the actual batch size is = micro_batch_size x gradient_accum_steps
    micro_batch_size: int = 2
    gradient_accum_steps: int = 64
    val_interval: int = 500
    val_iters: int = 400  # large size since micro_batch_size is very small
    log_interval: int = 100  # log training metrics (loss, accuracy)
    ckpt_interval: int = 500  # save model checkpoints every N training iterations

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    train_bias: str = 'none'  # none, lora_only, all
    train_head: str = 'lm_head'  # none, lm_head, the performance is not great when also train head

    # learning rate scheduler
    init_lr: float = 1e-6  # use a much smaller initial learning rate
    max_lr: float = 4e-5  # max learning rate when warm up
    min_lr: float = 2e-5  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 2000

    # prompt is less important than completion
    prompt_loss_weight: float = 0.01
    completion_loss_weight: float = 1.0

    # Adam optimizer
    use_bnb_8bit: bool = False  # use bitsandbytes 8bit optimizer
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_fused: bool = True  # only applicable if not using bitsandbytes 8bit optimizer

    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # BUG in torch 2.0.1

    # others
    seed: int = 127
    log_dir: str = './logs/finetune_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/finetune_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
