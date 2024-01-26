# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple
from dataclasses import dataclass


@dataclass
class config:
    """Trains Reward Model (RM) using LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '3B'  # 7B, 13B, 70B

    # RM model is special because we need to maintain >=2 graphs when computing loss, which requires more GPU RAM
    max_seq_len: int = 512

    rm_ckpt_file: str = './checkpoints/7b-sft/steps-2200-merged.pth'  # load fine-tuned checkpoint
    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    random_head_weights: bool = True

    # datasets
    train_datasources: Tuple[str] = (
        # './datasets/stack_exchange_preferences/train.pkl',  # this demands more GPU RAM because we have more than 2 responses per sample
        './datasets/hh-rlhf/train.pkl',
    )
    val_datasources: Tuple[str] = (
        # './datasets/stack_exchange_preferences/validation.pkl',
        './datasets/hh-rlhf/validation.pkl',
    )
    dataloader_workers: int = 1

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    num_epochs: int = 5
    # we always use micro batch size of 1 (sample) during training and evaluation
    gradient_accum_steps: int = 128
    loss_scale: float = 1.0 / 16  # scale loss to account for gradient accumulation, we don't want to use a very small scale
    val_interval: int = 200
    val_steps: int = 200
    log_interval: int = 5  # log training metrics (loss, accuracy)
    ckpt_interval: int = 200  # save model checkpoints every N training iterations

    # normalize and clip reward before compute loss during training and validation
    normalize_reward: bool = True
    max_abs_reward: float = 0.0

    # LoRA configuration
    lora_r: int = 128
    lora_scaling: float = 1.0  # set the LoRA scaling, by default 1.0 no scaling
    lora_dropout: float = 0.0

    # LoRA trainable layers
    lora_attn_query: bool = True  # train Attention query layer
    lora_attn_key: bool = True  # train Attention key layer
    lora_attn_value: bool = True  # train Attention value layer
    lora_attn_proj: bool = True  # train Attention projection layer
    lora_attn_mlp: bool = True  # train Attention MLP block

    train_bias: str = 'all'  # none, lora_only, all
    train_head: bool = True  # note we don't apply LoRA to model output head

    # Quantization
    quant_4bit: bool = False  # quantize frozen linear layer
    quant_lora_4bit: bool = False  # quantize LoRA linear layer
    quant_4bit_double: bool = True  # double quantize
    quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'

    # learning rate, maybe use smaller lr if also train head since we don't apply LoRA head layer
    init_lr: float = 3e-5  # initial learning rate
    max_lr: float = 3e-4  # max learning rate after warm up
    min_lr: float = 3e-4  # min learning rate after decay
    warmup_ratio: float = 0.03

    # AdamW optimizer
    use_paged_adamw: bool = False
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = False  # only applicable if not using bitsandbytes optimizer
    grad_clip: float = 0.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1

    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # not working with QLoRA

    # others
    seed: int = 143
    log_dir: str = './logs/rm_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/rm_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
