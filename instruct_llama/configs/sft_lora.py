# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class config:
    """Supervised fine-tuning using LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    max_seq_len: int = 512

    pretrain_ckpt_file: str = '/home/michael/models/meta_llama2/llama-2-7b/consolidated.pth'  # load pretrained checkpoint
    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = (
        './datasets/alpaca/train.pkl',
        './datasets/hh_rlhf_finetune/train.pkl',  # 120k
        # './datasets/stack_exchange_finetune/train.pkl',
        # './datasets/dolly/train.pkl',
        # './datasets/squad/train.pkl',
        # './datasets/commonsense_dialogues/train.pkl',
        # './datasets/deepmind_mathematics/train.pkl',
    )
    val_datasources: Tuple[str] = (
        './datasets/alpaca/validation.pkl',
        './datasets/hh_rlhf_finetune/validation.pkl',
        # './datasets/stack_exchange_finetune/validation.pkl',
        # './datasets/dolly/validation.pkl',
        # './datasets/squad/validation.pkl',
        # './datasets/commonsense_dialogues/validation.pkl',
        # './datasets/deepmind_mathematics/validation.pkl',
    )
    dataloader_workers: int = 1

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    num_epochs: int = 2
    # accumulate gradients so for each iteration, the actual batch size is = train_batch_size x gradient_accum_steps
    train_batch_size: int = 3
    gradient_accum_steps: int = 12
    val_interval: int = 500
    val_batch_size: int = 30
    val_steps: int = 20
    log_interval: int = 5  # log training metrics (loss, accuracy)
    ckpt_interval: int = 500  # save model checkpoints every N Training steps

    # LoRA configuration
    lora_r: int = 128
    lora_scaling: float = 1.0  # set the LoRA scaling, by default 1.0 no scaling
    lora_dropout: float = 0.05

    # LoRA trainable layers
    lora_attn_query: bool = True  # train Attention query layer
    lora_attn_key: bool = False  # train Attention key layer
    lora_attn_value: bool = True  # train Attention value layer
    lora_attn_proj: bool = False  # train Attention projection layer
    lora_attn_mlp: bool = False  # train Attention MLP block
    lora_head: bool = False  # train model output layer

    train_bias: str = 'none'  # none, lora_only, all

    # Quantization
    quant_4bit: bool = True  # quantize frozen linear layer
    quant_lora_4bit: bool = False  # quantize LoRA linear layer
    quant_4bit_double: bool = True  # double quantize
    quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'

    # learning rate, should use smaller lr if also train lm head since we don't apply LoRA to the head layer
    init_lr: float = 1e-5  # initial learning rate
    max_lr: float = 1e-4  # max learning rate after warm up
    min_lr: float = 1e-5  # min learning rate after decay
    warmup_ratio: float = 0.02

    # prompt is less important than completion
    prompt_loss_weight: float = 0.0
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    use_paged_adamw: bool = False
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.999)
    adam_eps: float = 1e-5
    adam_fused: bool = True  # only applicable if not using bitsandbytes optimizer
    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0

    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.

    # others
    seed: int = 127
    log_dir: str = './logs/sft_lora'
    ckpt_dir: str = './checkpoints/sft_lora'
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
