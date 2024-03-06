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

from instruct_llama.core.custom_dataset import DataSource


@dataclass
class config:
    """Trains policy and value networks using PPO (RL)"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    policy_model_type: str = '7B'  # 7B, 13B, 70B
    value_model_type: str = '3B'  # 3B, 7B, 13B, 70B
    reward_model_type: str = '3B'  # 3B, 7B, 13B, 70B
    max_seq_len: int = 400

    ref_ckpt_file: str = './checkpoints/sft/7B-steps-6000.pth'  # load fine-tuned checkpoint for reference model
    reward_ckpt_file: str = './checkpoints/rm/3B-steps-5000.pth'  # load RM checkpoint
    policy_ckpt_file: str = './checkpoints/sft/7B-steps-6000.pth'  # default same as fine-tuned checkpoint
    value_ckpt_file: str = './checkpoints/rm/3B-steps-5000.pth'  # default same as RM checkpoint
    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    # datasets
    train_prompt_datasources: Tuple[str] = (
        './datasets/hh_rlhf_prompt_only/train.pkl',
        # './datasets/stack_exchange_prompt_only/train.pkl',
    )
    val_prompt_datasources: Tuple[str] = (
        './datasets/hh_rlhf_prompt_only/validation.pkl',
        # './datasets/stack_exchange_prompt_only/validation.pkl',
    )

    dataloader_workers: int = 1
    max_train_samples: int = 20000  # set sample limit in the prompt dataset for a quick test run, 0 means no limit
    max_val_samples: int = 2000
    min_prompt_len: int = 16  # limit the minimum length of prompts in the prompt dataset
    max_prompt_len: int = 200  # limit the maximum length of prompts in the prompt dataset

    # model devices
    reward_device: str = 'cuda:0'  # RM model, frozen
    ref_device: str = 'cuda:0'  # SFT model, frozen
    policy_device: str = 'cuda:0'  # PPO policy model
    value_device: str = 'cuda:0'  # PPO value model

    # RL agent selfplay
    selfplay_batch_size: int = 32  # how many prompts to work on during selfplay to generate training samples in a single batch
    train_temperature: float = 1.0  # temperature, top_k, and top_p control exploration level during RL selfplay
    train_top_k: float = 0.0
    train_top_p: float = 1.0
    val_temperature: float = 0.7
    val_top_k: float = 0.0
    val_top_p: float = 0.9
    min_gen_len: int = 2  # episode with completion tokens lesser than this are discarded
    max_gen_len: int = 200  # cut maximum completion tokens in the batch, this does not necessarily apply to every episode
    clip_reward: float = 2.0  # if > 0, clip (normalized) environment reward in the range of [-clip_reward, clip_reward]
    selfplay_log_interval: int = 5  # frequency to log episode metrics (reward, steps etc.) to tensorboard
    selfplay_sample_interval: int = 100  # frequency to log selfplay generated text to tensorboard

    # PPO learning
    max_episodes: int = int(5e5)  # total number of training episodes
    train_rollout_size: int = 256  # how many training selfplay episodes to generate per epoch
    ppo_batch_size: int = 32
    ppo_update_epochs: int = 4  # PPO update epoch
    policy_micro_bs: int = 2  # micro batch size for policy
    value_micro_bs: int = 8  # micro batch size for value, we're using a much smaller value network than policy
    train_log_interval: int = 5  # log training metrics (loss etc.)
    discount: float = 1.0
    gae_lambda: float = 0.95
    policy_clip_eps: float = 0.2
    entropy_coef: float = 0.0
    whiten_advantages: bool = True

    # InstructGPT specific
    value_clip_eps: float = 0.2
    ptx_coef: float = 0.0  # 0.05 pre-training loss coefficient
    # pre-training datasets is optional if ptx_coef = 0
    ptx_datasources: Tuple[DataSource] = None
    # ptx_datasources: Tuple[DataSource] = (
    #     DataSource(
    #         name='red_pajama_mini',
    #         weights=1.0,
    #         data_file='./datasets/red_pajama_mini/train.npy',
    #         metadata_file='./datasets/red_pajama_mini/train_meta.json',
    #     ),
    # )
    scale_kl: bool = False  # remove negative KL and scale into [0, 1] before adding as penalties to reward
    # adaptive KL
    init_kl_coef: float = 0.05  # coefficient for per-token KL penalties
    adaptive_kl: bool = True
    kl_target: float = 6.0  # KL target estimate is summed up over a single episode then averaged over batch
    adaptive_kl_horizon: int = 10000  # number of selfplay episodes
    whiten_rewards: bool = False  # normalize combined reward (env_reward + KL penalties) before compute GAE advantages

    # looking for the first occurrence of truncate token text like '[INST]' in the response, so we may add penalty reward to the sequence
    truncate_token: Optional[Tuple[str]] = None  # ('[INST]', '[/INST]')
    truncate_penalty_value: float = 0.0  # add a negative value (like -1) for the truncated sequence to the response

    # validation
    var_interval: int = 20  # N epochs, which is around N x train_rollout_size episodes
    val_rollout_size: int = 128
    ckpt_interval: int = 20  # N epochs, which is around N x train_rollout_size episodes

    # frozen the first N decoder layers and make the last M-N decoder layers along with the output layer fully-trainable
    policy_frozen_layers: int = 26
    value_frozen_layers: int = 10

    # AdamW optimizer
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = False

    # PPO policy model learning rate, should use smaller lr if also train lm head since we don't apply LoRA to the head layer
    policy_init_lr: float = 4.5e-6  # initial learning rate
    policy_max_lr: float = 4.5e-6  # max learning rate after warm up
    policy_min_lr: float = 4.5e-6  # min learning rate after decay
    policy_lr_warmup_steps: int = 0
    policy_grad_clip: float = 5.0

    # PPO value model learning rate, should use smaller lr if also train scalar head since we don't apply LoRA to the head layer
    value_init_lr: float = 9e-6  # initial learning rate
    value_max_lr: float = 9e-6  # max learning rate after warm up
    value_min_lr: float = 9e-6  # min learning rate after decay
    value_lr_warmup_steps: int = 0
    value_grad_clip: float = 10.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0

    gradient_checkpointing: bool = False

    # others
    seed: int = 152
    log_dir: str = './logs/rlhf'
    ckpt_dir: str = './checkpoints/rlhf'
