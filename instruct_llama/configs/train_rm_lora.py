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

    max_seq_len: int = 320  # RM model is special because we need to maintain two graphs when computing loss, so the sequence length is much shorter
    # minimum number of completions per sample, if completions lesser than this, the entire sample is discarded
    min_completions: int = 2
    # maximum number of completions per sample, if completions more than this, only keep the best N samples, set this to a lower value if don't have enough GPU RAM
    max_completions: int = 4

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we will soon found out if CUDA out of memory error occurs, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    # 10000 training samples / 32 = 300 iters for one epoch
    max_train_iters: int = 1500
    # we always use batch size of 1 when sample from the dataset,
    # this is measured over number of sample, note one sample might have 2 to N completions
    gradient_accum_steps: int = 32
    # if one sample have more than 2 completions, will break them into smaller pieces when computing the rewards
    micro_batch_size: int = 2
    val_interval: int = 300
    val_iters: int = 200  # large size since micro_batch_size is very small
    log_interval: int = 100  # log training metrics (loss, accuracy)
    ckpt_interval: int = 300  # save model checkpoints every N training iterations

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    train_bias: str = 'none'  # none, lora_only, all
    train_head: str = 'scalar_head'  # none, scalar_head, train head is required since the output layer is initialized randomly

    # learning rate scheduler
    init_lr: float = 2e-5  # use a much smaller initial learning rate
    max_lr: float = 2e-5  # max learning rate when warm up
    min_lr: float = 2e-5  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 1500

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
    log_dir: str = './logs/train_rm_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/train_rm_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
