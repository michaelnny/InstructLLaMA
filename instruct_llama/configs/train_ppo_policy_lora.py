from typing import Tuple
from dataclasses import dataclass


@dataclass
class config:
    """Trains policy using PPO (RL) and LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    head_type: str = 'lm_and_scalar_heads'  # save GPU RAM by using a single model with shared weights

    sft_ckpt_file: str = './checkpoints/7b-finetune/iter-2000-merged.pth'  # load fine-tuned checkpoint
    rm_ckpt_file: str = './checkpoints/7b-rm/iter-3200-merged.pth'  # load RM checkpoint
    tokenizer_file: str = './meta_checkpoints/llama-2/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = (
        # './datasets/hh-rlhf_red_team_prompt_only/train.pkl',
        './datasets/hh-rlhf_prompt_only/train.pkl',
    )
    val_datasources: Tuple[str] = (
        # './datasets/hh-rlhf_red_team_prompt_only/validation.pkl',
        './datasets/hh-rlhf_prompt_only/validation.pkl',
    )
    dataloader_workers: int = 1

    max_seq_len: int = 600  # use smaller sequence length to save GPU RAM

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    # 12500 training samples / 32 = 400 iters for one epoch
    max_train_iters: int = 4000
    # we always use batch size of 1 when sample from the dataset,
    # this is measured over number of sample, note one sample might have 2 to N completions
    gradient_accum_steps: int = 32
    val_interval: int = 100
    val_iters: int = 200  # large size since micro_batch_size is very small
    log_interval: int = 20  # log training metrics (loss, accuracy)
    ckpt_interval: int = 400  # save model checkpoints every N training iterations

    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    train_bias: str = 'none'  # none, lora_only, all
    train_head: bool = True  # train lm_head and scalar_head

    # AdamW optimizer
    use_bnb_8bit: bool = True  # use bitsandbytes 8bit optimizer
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_fused: bool = True  # only applicable if not using bitsandbytes 8bit optimizer
    lr: float = 2e-5

    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # BUG in torch 2.0.1

    # others
    seed: int = 127
    log_dir: str = './logs/train_ppo_policy_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/train_ppo_policy_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
