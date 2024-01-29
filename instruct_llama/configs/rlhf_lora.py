# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple
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
    reward_model_type: str = '3B'  # 3B, 7B, 13B, 70B

    max_seq_len: int = 512

    sft_ckpt_file: str = './checkpoints/7b-sft/steps-2200-merged.pth'  # load fine-tuned checkpoint
    rm_ckpt_file: str = './checkpoints/3b-rm/steps-2200-merged.pth'  # load RM checkpoint
    rm_norm_ckpt_file: str = './checkpoints/3b-rm/normalizer_3B-steps-2200.pth'  # load RM normalizer checkpoint
    policy_ckpt_file: str = './checkpoints/7b-sft/steps-2200-merged.pth'  # default same as fine-tuned checkpoint
    value_ckpt_file: str = './checkpoints/3b-rm/steps-2200-merged.pth'  # default same as RM checkpoint

    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

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

    train_prompt_datasources: Tuple[str] = ('./datasets/hh-rlhf_prompt_only/train.pkl',)
    val_prompt_datasources: Tuple[str] = ('./datasets/hh-rlhf_prompt_only/validation.pkl',)

    dataloader_workers: int = 1
    max_train_samples: int = 100000  # limit the amount of sample prompts to train the agent
    max_val_samples: int = 10000  # limit the amount of sample prompts to validate the agent
    min_prompt_len: int = 12  # limit the minimum length of prompts in the prompt dataset
    max_prompt_len: int = 200  # limit the maximum length of prompts in the prompt dataset

    # model devices
    env_device: str = 'cuda:0'  # RM model and SFT model, these are frozen
    policy_device: str = 'cuda:0'  # PPO policy model
    value_device: str = 'cuda:0'  # PPO value model

    # RL agent selfplay
    selfplay_batch_size: int = 26  # how many prompts to work on during selfplay to generate training samples
    train_temperature: float = 1.0
    train_top_p: float = 1.0
    val_temperature: float = 0.7
    val_top_p: float = 0.9
    min_gen_len: int = 6  # episode with completion tokens lesser than this are discarded
    selfplay_log_interval: int = 20  # log episode metrics (reward, steps etc.)
    normalize_env_rewards: bool = True
    clip_env_reward: float = 1.0  # clip (normalized) environment reward in the range of [-max_abs_reward, max_abs_reward]

    # PPO learning
    max_episodes: int = int(1e6)  # total number of training episodes
    train_episodes_per_epoch: int = 512  # how many training selfplay episodes to run per epoch
    train_batch_size: int = 2
    gradient_accum_steps: int = 64
    loss_scale: float = 1.0 / 32  # scale loss to account for gradient accumulation, we don't want to use a very small scale
    train_log_interval: int = 5  # log training metrics (loss etc.)
    update_epochs: int = 4  # PPO update epoch
    discount: float = 1.0
    gae_lambda: float = 0.95
    policy_clip_eps: float = 0.2
    entropy_coef: float = 0.0
    normalize_advantages: bool = True

    # InstructGPT specific
    value_clip_eps: float = 0.0
    ptx_coef: float = 0.0  # pre-training loss coefficient
    kl_coef: float = 0.02  # per-token KL penalties loss coefficient
    clip_kl: float = 0.25  # clip per-token KL penalties in the range of [-max_abs_kl, max_abs_kl]
    normalize_rewards: bool = False  # normalize combined reward (env_reward - kl)

    # validation
    var_interval: int = 5  # N epochs
    val_episodes_per_epoch: int = 256
    ckpt_interval: int = 5  # N epochs

    # LoRA configuration
    lora_r: int = 128
    lora_scaling: float = 1.0  # set the LoRA scaling, by default 1.0 no scaling
    lora_dropout: float = 0.0

    # LoRA trainable layers
    lora_attn_query: bool = True  # train Attention query layer
    lora_attn_key: bool = False  # train Attention key layer
    lora_attn_value: bool = True  # train Attention value layer
    lora_attn_proj: bool = False  # train Attention projection layer
    lora_attn_mlp: bool = True  # train Attention MLP block

    train_bias: str = 'all'  # none, lora_only, all
    train_head: bool = True  # note we don't apply LoRA to model output head

    # Quantization
    quant_4bit: bool = True  # quantize frozen linear layer
    quant_lora_4bit: bool = True  # quantize LoRA linear layer
    quant_4bit_double: bool = True  # double quantize
    quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'

    # AdamW optimizer
    use_paged_adamw: bool = False
    weight_decay: float = 0.0
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-5
    adam_fused: bool = True  # only applicable if not using bitsandbytes optimizer

    # PPO policy model
    policy_init_lr: float = 1e-5  # initial learning rate
    policy_max_lr: float = 5e-5  # max learning rate after warm up
    policy_min_lr: float = 1e-5  # min learning rate after decay
    policy_warmup_steps: int = 100
    policy_grad_clip: float = 5.0

    # PPO value model
    value_init_lr: float = 5e-5  # initial learning rate
    value_max_lr: float = 8e-5  # max learning rate after warm up
    value_min_lr: float = 5e-5  # min learning rate after decay
    value_warmup_steps: int = 100
    value_grad_clip: float = 0.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0

    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # not working with QLoRA

    # others
    seed: int = 152
    log_dir: str = './logs/rlhf_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/rlhf_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
