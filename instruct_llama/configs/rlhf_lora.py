# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple, Optional
from dataclasses import dataclass

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from instruct_llama.utils.custom_dataset import DataSource


@dataclass
class config:
    """Trains policy and value networks using PPO (RL) and LoRA"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    policy_model_type: str = '7B'  # 7B, 13B, 70B
    value_model_type: str = '3B'  # 3B, 7B, 13B, 70B
    reward_model_type: str = '3B'  # 3B, 7B, 13B, 70B
    max_seq_len: int = 512

    sft_ckpt_file: str = './checkpoints/7b-sft/steps-2200-merged.pth'  # load fine-tuned checkpoint
    rm_ckpt_file: str = './checkpoints/3b-rm/steps-8000-merged.pth'  # load RM checkpoint
    rm_normalizer_ckpt_file: str = None  # load RM normalizer checkpoint
    policy_ckpt_file: str = './checkpoints/7b-sft/steps-2200-merged.pth'  # default same as fine-tuned checkpoint
    value_ckpt_file: str = './checkpoints/3b-rm/steps-8000-merged.pth'  # default same as RM checkpoint
    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    # datasets
    # pre-training datasets is optional if ptx_coef = 0
    train_ptx_datasources: Tuple[DataSource] = None
    # train_ptx_datasources: Tuple[DataSource] = (
    #     DataSource(
    #         name='red_pajama_mini',
    #         weights=1.0,
    #         data_file='./datasets/red_pajama_mini/train.npy',
    #         metadata_file='./datasets/red_pajama_mini/train_meta.json',
    #     ),
    # )

    train_prompt_datasources: Tuple[str] = (
        './datasets/hh_rlhf_prompt_only/train.pkl',
        # './datasets/stack_exchange_prompt_only/train.pkl',
    )
    val_prompt_datasources: Tuple[str] = (
        './datasets/hh_rlhf_prompt_only/validation.pkl',
        # './datasets/stack_exchange_prompt_only/validation.pkl',
    )

    dataloader_workers: int = 1
    max_train_samples: int = 10000  # set sample limit in the dataset for a quick test run, 0 means no limit
    max_val_samples: int = 1000
    min_prompt_len: int = 32  # limit the minimum length of prompts in the prompt dataset
    max_prompt_len: int = 200  # limit the maximum length of prompts in the prompt dataset

    # model devices
    env_device: str = 'cuda:0'  # RM model and SFT model, these are frozen
    policy_device: str = 'cuda:0'  # PPO policy model
    value_device: str = 'cuda:0'  # PPO value model

    # RL agent selfplay
    selfplay_batch_size: int = 32  # how many prompts to work on during selfplay to generate training samples
    train_temperature: float = 0.7
    train_top_p: float = 0.9
    val_temperature: float = 0.7
    val_top_p: float = 0.9
    min_gen_len: int = 4  # episode with completion tokens lesser than this are discarded
    normalize_env_rewards: bool = True  # normalized RM reward to have zero mean and unit variance
    clip_env_reward: float = 1.0  # clip (normalized) environment reward in the range of [-max_abs_reward, max_abs_reward]
    selfplay_log_interval: int = 10  # frequency log episode metrics (reward, steps etc.)
    selfplay_sample_interval: int = 100  # frequency to log selfplay generated text

    # PPO learning
    max_episodes: int = int(1e6)  # total number of training episodes
    rollout_size: int = 128  # how many training selfplay episodes to generate per epoch
    train_batch_size: int = 2
    gradient_accum_steps: int = 16
    loss_scale: float = 1.0 / 4  # scale loss to account for gradient accumulation, we don't want to use a very small scale
    train_log_interval: int = 5  # log training metrics (loss etc.)
    ppo_update_epochs: int = 4  # PPO update epoch
    discount: float = 1.0
    gae_lambda: float = 0.95
    policy_clip_eps: float = 0.2
    entropy_coef: float = 0.0
    whiten_advantages: bool = True

    # InstructGPT specific
    value_clip_eps: float = 0.2
    ptx_coef: float = 0.0  # pre-training loss coefficient
    init_kl_coef: float = 0.03  # 0.15  # per-token KL penalties coefficient
    adaptive_kl: bool = False
    kl_target: float = 6.0  # target kl summed up over a single episode then averaged over batch
    adaptive_kl_horizon: int = 10000  # number of selfplay episodes
    whiten_rewards: bool = False  # normalize combined reward (env_reward - kl)

    # looking for the first occurrence of tokens like '[INST]' in the response, so we may add penalty to the reward
    truncate_token: Optional[Tuple[str]] = None  # ('[INST]', '[/INST]')
    truncate_penalty_value: float = 0.0  # add a negative value (like -1) for the truncated sequence to the response

    # validation
    var_interval: int = 20  # N epochs, which is around N x rollout_size episodes
    val_episodes_per_epoch: int = 64
    ckpt_interval: int = 20  # N epochs, which is around N x rollout_size episodes

    # LoRA configuration
    lora_r: int = 128
    lora_scaling: float = 1.0  # set the LoRA scaling, by default 1.0 no scaling
    lora_dropout: float = 0.0

    # LoRA trainable layers
    lora_attn_query: bool = True  # train Attention query layer
    lora_attn_key: bool = False  # train Attention key layer
    lora_attn_value: bool = True  # train Attention value layer
    lora_attn_proj: bool = False  # train Attention projection layer
    lora_attn_mlp: bool = False  # train Attention MLP block

    train_bias: str = 'all'  # none, lora_only, all
    train_head: bool = True  # note we don't apply LoRA to model output head

    # Quantization
    quant_4bit: bool = True  # quantize frozen linear layer
    quant_lora_4bit: bool = False  # quantize LoRA linear layer
    quant_4bit_double: bool = True  # double quantize
    quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'

    # AdamW optimizer
    use_paged_adamw: bool = False
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = True  # only applicable if not using bitsandbytes optimizer

    # PPO policy model learning rate
    policy_init_lr: float = 2.5e-5  # initial learning rate
    policy_max_lr: float = 2.5e-5  # max learning rate after warm up
    policy_min_lr: float = 2.5e-5  # min learning rate after decay
    policy_lr_warmup_steps: int = 0
    policy_grad_clip: float = 5.0

    # PPO value model learning rate
    value_init_lr: float = 5e-5  # initial learning rate
    value_max_lr: float = 5e-5  # max learning rate after warm up
    value_min_lr: float = 5e-5  # min learning rate after decay
    value_lr_warmup_steps: int = 0
    value_grad_clip: float = 50.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0

    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.

    # others
    seed: int = 152
    log_dir: str = './logs/rlhf_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/rlhf_lora'
