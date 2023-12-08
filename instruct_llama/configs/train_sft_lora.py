from typing import Tuple
from dataclasses import dataclass


@dataclass
class config:
    """Supervised fine-tuning uisng LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '1B'  # 7B, 13B, 70B
    head_type: str = 'lm_head'
    max_seq_len: int = 450  # use smaller sequence length to save GPU RAM

    pretrain_ckpt_file: str = './meta_checkpoints/llama-2/llama-2-7b/consolidated.pth'  # load pretrained checkpoint
    tokenizer_file: str = './meta_checkpoints/llama-2/tokenizer.model'  # load tokenizer model

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

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    # training samples * epochs / batch size, 70000 training samples, with batch size of 128, 500 iters = one epoch
    max_train_iters: int = 500 * 10
    # accumulate gradients so for each iteration, the actual batch size is = micro_batch_size x gradient_accum_steps
    micro_batch_size: int = 2
    gradient_accum_steps: int = 64
    loss_scale: float = 1 / 64
    val_interval: int = 500
    val_batch_size: int = 24
    val_iters: int = 100
    log_interval: int = 10  # log training metrics (loss, accuracy)
    ckpt_interval: int = 500  # save model checkpoints every N training iterations

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    train_bias: str = 'none'  # none, lora_only, all
    train_head: bool = True  # optional

    # learning rate scheduler, if also train head, should use a smaller learning rate
    init_lr: float = 1e-6  # use a much smaller initial learning rate
    max_lr: float = 5e-6  # max learning rate when warm up
    min_lr: float = 1e-6  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 500 * 5

    # prompt is less important than completion
    prompt_loss_weight: float = 0.01
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    use_bnb_8bit: bool = False  # use bitsandbytes 8bit optimizer
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = True  # only applicable if not using bitsandbytes 8bit optimizer
    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # BUG in torch 2.0.1 and 2.1.0, UserWarning: Torchinductor does not support code generation for complex operators. Performance may be worse than eager

    # others
    seed: int = 127
    log_dir: str = './logs/train_sft_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/train_sft_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
