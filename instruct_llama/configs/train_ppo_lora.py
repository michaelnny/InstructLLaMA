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
    max_seq_len: int = 450  # use smaller sequence length to save GPU RAM

    sft_ckpt_file: str = './checkpoints/7b-finetune/iter-2000-merged.pth'  # load fine-tuned checkpoint
    rm_ckpt_file: str = './checkpoints/7b-rm/iter-3200-merged.pth'  # load RM checkpoint
    tokenizer_file: str = './meta_checkpoints/llama-2/tokenizer.model'  # load tokenizer model

    # datasets
    pretrain_datasources: Tuple[DataSource] = (
        DataSource(
            name='red_pajama',
            weights=1.0,
            data_file='./datasets/red_pajama_mini/train.npy',
            metadata_file='./datasets/red_pajama_mini/train_meta.json',
        ),
    )
    dataloader_workers: int = 1

    prompt_datasources: Tuple[str] = (
        # './datasets/hh-rlhf_red_team_prompt_only/train.pkl',
        './datasets/hh-rlhf_prompt_only/train.pkl',
    )
    max_train_samples: int = 50000  # limit the amount of sample prompts to use during selfplay
    max_prompt_len: int = 256  # limit the maximum length of prompts

    # PPO actor selfplay
    selfplay_batch_size: int = 32  # how many prompts to work on during selfplay to generate training samples
    selfplay_temperature: float = 1.0
    selfplay_top_p: float = 0.9

    # PPO learning
    micro_batch_size: int = 2
    gradient_accum_steps: int = 32  # batch size of 64 episodes
    max_episodes: int = 1000000
    learn_interval: int = 64 * 8  # how many episodes to play before do learning
    train_log_interval: int = 5  # log training metrics (loss etc.)
    selfplay_log_interval: int = 10  # log episode metrics (reward, time etc.)
    ckpt_interval: int = 100
    update_epochs: int = 4
    discount: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    kl_coef: float = 0.02
    ptx_coef: float = 0.9
    value_coef: float = 0.1
    entropy_coef: float = 0.0
    grad_clip: float = 1.0
    normalize_reward: bool = False
    normalize_advantage: bool = True

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
    init_lr: float = 5e-7
    max_lr: float = 5e-6
    warmup_steps: int = 500

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # BUG in torch 2.0.1

    # others
    seed: int = 127
    log_dir: str = './logs/train_ppo_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/train_ppo_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
