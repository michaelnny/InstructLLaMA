# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class config:
    """Supervised fine-tuning using full-scale training"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    max_seq_len: int = 512

    pretrain_ckpt_file: str = '/home/michael/models/meta_llama2/llama-2-7b/consolidated.pth'  # load pretrained checkpoint
    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = (
        # './datasets/alpaca/train.pkl',
        # './datasets/dolly/train.pkl',
        # './datasets/squad/train.pkl',
        # './datasets/deepmind_mathematics/train.pkl',
        # './datasets/commonsense_dialogues/train.pkl',
        './datasets/hh_rlhf_finetune/train.pkl',  # train reference policy for RLHF training
    )
    val_datasources: Tuple[str] = (
        # './datasets/alpaca/validation.pkl',
        # './datasets/dolly/validation.pkl',
        # './datasets/squad/validation.pkl',
        # './datasets/deepmind_mathematics/validation.pkl',
        # './datasets/commonsense_dialogues/validation.pkl',
        './datasets/hh_rlhf_finetune/validation.pkl',
    )
    dataloader_workers: int = 1

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    num_epochs: int = 5
    # accumulate gradients so for each iteration, the actual batch size is = train_batch_size x gradient_accum_steps
    train_batch_size: int = 2
    gradient_accum_steps: int = 16
    val_interval: int = 500
    val_steps: int = 20
    val_batch_size: int = 30
    log_interval: int = 5  # log training metrics (loss, accuracy)
    ckpt_interval: int = 500  # save model checkpoints every N Training steps

    # frozen the first N decoder layers and make the last M-N decoder layers along with the output layer fully-trainable
    frozen_layers: int = 16
    train_atten_qv_layers_only: bool = True  # if true, only train the attention key and value layers, otherwise the entire encoder block

    # learning rate, should use smaller lr if also train lm head since we don't apply LoRA to the head layer
    init_lr: float = 9.65e-6  # initial learning rate
    max_lr: float = 9.65e-6  # max learning rate after warm up
    min_lr: float = 9.65e-7  # min learning rate after decay
    warmup_ratio: float = 0.0

    # prompt is less important than completion
    prompt_loss_weight: float = 0.01
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    use_paged_adamw: bool = False
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = True  # only applicable if not using bitsandbytes optimizer
    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    head_dropout: float = 0.1

    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.

    # others
    seed: int = 127
    log_dir: str = './logs/sft'
    ckpt_dir: str = './checkpoints/sft'
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
