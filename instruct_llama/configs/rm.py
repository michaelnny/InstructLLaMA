# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class config:
    """Trains Reward Model (RM)"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '3B'  # 3B, 7B, 13B, 70B
    max_seq_len: int = 512

    # load fine-tuned checkpoint or previously trained RM model checkpoint
    reward_ckpt_file: str = './checkpoints/sft/7B-steps-6000.pth'
    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    random_head_weights: bool = True  # remove this in case resume training

    # datasets
    train_datasources: Tuple[str] = (
        './datasets/hh_rlhf_comparison/train.pkl',
        # './datasets/stack_exchange_comparison/train.pkl',  # this demands more GPU RAM because we have more than 2 responses per sample
    )
    val_datasources: Tuple[str] = (
        './datasets/hh_rlhf_comparison/validation.pkl',
        # './datasets/stack_exchange_comparison/validation.pkl',
    )
    dataloader_workers: int = 1

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    num_epochs: int = 3
    # this is number of sample, the actual batch for forward pass might be larger since one sample could have >=2 responses
    train_batch_size: int = 4
    gradient_accum_steps: int = 8
    val_interval: int = 500
    val_steps: int = 40
    val_batch_size: int = 30
    log_interval: int = 5  # log training metrics (loss, accuracy)
    ckpt_interval: int = 500  # save model checkpoints every N Training steps

    # number of samples to collect statistics for reward normalizer after training is done
    norm_samples: int = 10000
    norm_batch_size: int = 30

    # frozen the first N decoder layers and make the last M-N decoder layers along with the output layer fully-trainable
    frozen_layers: int = 10

    # learning rate, should use smaller lr since we're doing full-scale training
    init_lr: float = 1.2e-6  # initial learning rate
    max_lr: float = 1.2e-5  # max learning rate after warm up
    min_lr: float = 1.2e-6  # min learning rate after decay
    warmup_ratio: float = 0.02

    # AdamW optimizer
    use_paged_adamw: bool = False
    weight_decay: float = 0.002
    adam_betas: Tuple = (0.9, 0.999)
    adam_eps: float = 1e-5
    adam_fused: bool = True  # only applicable if not using bitsandbytes optimizer
    grad_clip: float = 10.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0

    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.

    # others
    seed: int = 143
    log_dir: str = './logs/rm'
    ckpt_dir: str = './checkpoints/rm'
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
