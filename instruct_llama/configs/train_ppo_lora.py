from typing import Tuple
from dataclasses import dataclass

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from instruct_llama.utils import DataSource


@dataclass
class config:
    """Trains policy using PPO (RL) and LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    max_seq_len: int = 430  # use smaller sequence length to save GPU RAM

    sft_ckpt_file: str = './checkpoints/7b-sft/iter-2000-merged.pth'  # load fine-tuned checkpoint
    rm_ckpt_file: str = './checkpoints/7b-rm/iter-2500-merged.pth'  # load RM checkpoint
    policy_ckpt_file: str = './checkpoints/7b-sft/iter-2000-merged.pth'  # default same as fine-tuned checkpoint
    value_ckpt_file: str = './checkpoints/7b-rm/iter-2500-merged.pth'  # default same as RM checkpoint
    tokenizer_file: str = './meta_checkpoints/llama-2/tokenizer.model'  # load tokenizer model

    # datasets
    train_ptx_datasources: Tuple[DataSource] = (
        DataSource(
            name='red_pajama_mini',
            weights=1.0,
            data_file='./datasets/red_pajama_mini/train.npy',
            metadata_file='./datasets/red_pajama_mini/train_meta.json',
        ),
    )
    val_ptx_datasources: Tuple[DataSource] = (
        DataSource(
            name='red_pajama_mini',
            weights=1.0,
            data_file='./datasets/red_pajama_mini/validation.npy',
            metadata_file='./datasets/red_pajama_mini/validation_meta.json',
        ),
    )

    train_prompt_datasources: Tuple[str] = (
        # './datasets/hh-rlhf_red_team_prompt_only/train.pkl',
        './datasets/hh-rlhf_prompt_only/train.pkl',
    )
    val_prompt_datasources: Tuple[str] = (
        # './datasets/hh-rlhf_red_team_prompt_only/validation.pkl',
        './datasets/hh-rlhf_prompt_only/validation.pkl',
    )

    dataloader_workers: int = 1
    max_train_samples: int = 1000  # limit the amount of sample prompts to train the agent
    max_val_samples: int = 100  # limit the amount of sample prompts to validate the agent
    max_prompt_len: int = 200  # limit the maximum length of prompts

    # RL actor selfplay
    selfplay_batch_size: int = 32  # how many prompts to work on during selfplay to generate training samples
    train_temperature: float = 1.0
    train_top_p: float = 1.0
    val_temperature: float = 0.7
    val_top_p: float = 0.9
    min_gen_len: int = 6  # episode with lesser completion tokens are discarded
    max_gen_len: int = 250
    selfplay_log_interval: int = 100  # log episode metrics (reward, steps etc.)

    # PPO learning
    max_episodes: int = 1000000  # total number of training episodes
    warmup_episodes: int = 128  # how many selfplay episodes to warm up reward norm statistics
    train_episodes_per_epoch: int = 512  # how many training selfplay episodes to run per epoch
    micro_batch_size: int = 2
    gradient_accum_steps: int = 32  # batch size of 2x32 episodes
    train_log_interval: int = 10  # log training metrics (loss etc.)
    update_epochs: int = 4
    discount: float = 1.0
    gae_lambda: float = 0.95
    policy_clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.0
    ptx_coef: float = 0.0
    kl_coef: float = 0.02

    normalize_env_rewards: bool = True

    clip_env_reward: float = 1.0  # clip (normalize) environment reward in the range of [-max_abs_reward, max_abs_reward]
    clip_kl: float = 0.0  # clip KL in the range of [-max_abs_kl, max_abs_kl]

    whiten_rewards: bool = True
    whiten_advantages: bool = True

    val_episodes_per_epoch: int = 100  # how many validation selfplay episodes to run per epoch
    var_interval: int = 5  # validation agent every N epochs

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    train_bias: str = 'none'  # none, lora_only, all
    train_head: bool = True  # train lm_head and scalar_head

    # AdamW optimizer
    use_bnb_8bit: bool = False  # use bitsandbytes 8bit optimizer
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = False  # only applicable if not using bitsandbytes 8bit optimizer

    policy_init_lr: float = 1e-6
    policy_max_lr: float = 1e-6
    policy_warmup_steps: int = 100
    policy_grad_clip: float = 1.0

    value_init_lr: float = 1e-6
    value_max_lr: float = 5e-6
    value_warmup_steps: int = 100
    value_grad_clip: float = 0.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # BUG in torch 2.0.1 and 2.1.0, UserWarning: Torchinductor does not support code generation for complex operators. Performance may be worse than eager

    # others
    seed: int = 125
    log_dir: str = './logs/train_ppo_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/train_ppo_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
