# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Train policy and value models using PPO algorithm (RL), starting from fine-tuned model and reward model (RM) checkpoints."""

import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model import Transformer, ModelArgs
from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.configs.rlhf import config as cfg
from instruct_llama.core.custom_dataset import BlendedDataset, PromptOnlyDataset
from instruct_llama.core.schedule import CosineDecayWithWarmupLRScheduler
from instruct_llama.core.train_helper import make_model_layer_trainable, create_optimizer, compute_num_trainable_params
from instruct_llama.core.rl_ppo import AdaptiveKLController, PPOAgent
from instruct_llama.utils.logger import create_logger
from instruct_llama.utils.checkpoint import create_checkpoint


logger = create_logger()


def convert_model_to_dtype(model: Transformer, compute_dtype: torch.dtype):
    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=compute_dtype)


def build_model(
    vocab_size: int,
    ckpt_file: str,
    compute_dtype,
    model_type: str,
    max_seq_len: int,
    max_batch_size: int = 1,
    embed_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    gradient_checkpointing: bool = False,
    frozen: bool = True,
    head_type: str = 'lm_head',
    strict: bool = True,
    device: str = 'cpu',
) -> Transformer:
    assert vocab_size > 0

    model_args = ModelArgs.from_model_type(
        model_type=model_type,
        vocab_size=vocab_size,
        # Regular configurations
        head_type=head_type,
        use_cache=False,
        max_seq_len=max_seq_len,
        max_batch_size=(max_batch_size if (head_type == 'dual_head' or head_type == 'lm_head') and not frozen else 1),
        embed_dropout=0.0 if frozen else embed_dropout,
        attn_dropout=0.0 if frozen else attn_dropout,
        gradient_checkpointing=False if frozen else gradient_checkpointing,
    )

    model = Transformer(model_args)

    if os.path.exists(ckpt_file):
        print(f'Loading model checkpoint {ckpt_file!r} ...')
        model_state = torch.load(ckpt_file)
        model.load_state_dict(model_state, strict=strict)
        del model_state  # free up CPU RAM

    convert_model_to_dtype(model, compute_dtype)

    if frozen:
        for p in model.parameters():
            p.requires_grad = False

    return model.to(device)


def clear_gpu_cache():
    torch.cuda.empty_cache()


def main():
    assert cfg.train_log_interval >= 1
    assert cfg.selfplay_log_interval >= 1
    assert cfg.min_gen_len >= 1
    assert cfg.max_gen_len >= 50
    assert cfg.max_prompt_len + cfg.max_gen_len <= cfg.max_seq_len
    assert cfg.ckpt_interval >= 1

    if cfg.reward_device == cfg.ref_device == cfg.policy_device == cfg.value_device or cfg.policy_device == cfg.value_device or cfg.reward_device == cfg.ref_device:
        logger.warning('Run models on the same device could cause CUDA OOM!!!!!!')
    if not os.path.exists(cfg.ref_ckpt_file):
        raise ValueError(f'Invalid SFT model checkpoint {cfg.ref_ckpt_file!r}, aborting ...')
    if not os.path.exists(cfg.reward_ckpt_file):
        raise ValueError(f'Invalid RM model checkpoint {cfg.reward_ckpt_file!r}, aborting ...')
    if not os.path.exists(cfg.policy_ckpt_file):
        raise ValueError(f'Invalid policy model checkpoint {cfg.policy_ckpt_file!r}, aborting ...')
    if not (torch.version.cuda and torch.cuda.is_bf16_supported()):
        raise RuntimeError('The script only supports training using CUDA and torch.bfloat16, but GPU does not support it.')
    if cfg.ptx_coef > 0 and cfg.ptx_datasources is None:
        raise ValueError('Invalid pre-training dataset, which is required when `ptx_coef > 0`, aborting ...')

    # --------------- Load datasets ---------------

    logger.info('Loading datasets ...')

    tokenizer = Tokenizer(cfg.tokenizer_file)
    vocab_size = tokenizer.vocab_size

    train_ptx_loader = None
    if cfg.ptx_datasources is not None:
        # Our custom IterableDatasets already have sharding and shuffle mechanism implemented
        cuda_kwargs = {
            'num_workers': cfg.dataloader_workers,
            'batch_size': cfg.policy_micro_bs,
            'pin_memory': False,
            'shuffle': False,
            'sampler': None,
        }
        train_ptx_dataset = BlendedDataset(
            data_sources=cfg.ptx_datasources,
            max_seq_len=cfg.max_seq_len,
            seed=cfg.seed,
        )

        train_ptx_loader = DataLoader(train_ptx_dataset, **cuda_kwargs)

        logger.info(f'PTX pretrain dataset metadata:\n{train_ptx_dataset.get_metadata()}')

    train_prompt_dataset = PromptOnlyDataset(
        data_sources=cfg.train_prompt_datasources,
        min_seq_len=cfg.min_prompt_len,
        max_seq_len=cfg.max_prompt_len,
        max_samples=cfg.max_train_samples,
    )

    logger.info(f'Train prompt dataset metadata:\n{train_prompt_dataset.get_metadata()}')

    val_prompt_dataset = PromptOnlyDataset(
        data_sources=cfg.val_prompt_datasources,
        min_seq_len=cfg.min_prompt_len,
        max_seq_len=cfg.max_prompt_len,
        max_samples=cfg.max_val_samples,
    )

    logger.info(f'Validation prompt dataset metadata:\n{val_prompt_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    compute_dtype = torch.bfloat16
    torch.set_default_dtype(compute_dtype)

    logger.info('Initializing reference and reward models ...')
    ref_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.ref_ckpt_file,
        compute_dtype=compute_dtype,
        model_type=cfg.policy_model_type,
        head_type='lm_head',
        max_seq_len=cfg.max_seq_len,
        frozen=True,
    )
    reward_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.reward_ckpt_file,
        compute_dtype=compute_dtype,
        model_type=cfg.reward_model_type,
        head_type='scalar_head',
        max_seq_len=cfg.max_seq_len,
        frozen=True,
    )

    logger.info('Initializing PPO policy and value models ...')

    # Load model checkpoint using strict=False,
    # because there are missing keys due to LoRA weights and the value head weights not contained in checkpoint state
    policy_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.policy_ckpt_file,
        compute_dtype=compute_dtype,
        model_type=cfg.policy_model_type,
        head_type='lm_head',
        max_seq_len=cfg.max_seq_len,
        max_batch_size=cfg.selfplay_batch_size,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
        frozen=False,
    )

    value_model = build_model(
        vocab_size=vocab_size,
        ckpt_file=cfg.value_ckpt_file,
        compute_dtype=compute_dtype,
        model_type=cfg.value_model_type,
        head_type='scalar_head',
        max_seq_len=cfg.max_seq_len,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
        frozen=False,
    )

    # freeze first N decoder layers and make last M-N decoder layers and output layer trainable
    policy_trainable_layers = ['post_norm', 'lm_head']
    value_trainable_layers = ['post_norm', 'scalar_head']
    for i in range(cfg.policy_frozen_layers, policy_model.n_layers):
        policy_trainable_layers.append(f'layers.{i}')
    for i in range(cfg.value_frozen_layers, value_model.n_layers):
        value_trainable_layers.append(f'layers.{i}')

    logger.info(f'PPO policy trainable layers:\n{policy_trainable_layers}')
    make_model_layer_trainable(policy_model, policy_trainable_layers)

    num_trainable, num_frozen = compute_num_trainable_params(policy_model)
    logger.info(f'PPO policy number of trainable parameters: {num_trainable:,}')
    logger.info(f'PPO policy number of frozen parameters: {num_frozen:,}')

    logger.info(f'PPO value trainable layers:\n{value_trainable_layers}')
    make_model_layer_trainable(value_model, value_trainable_layers)

    num_trainable, num_frozen = compute_num_trainable_params(value_model)
    logger.info(f'PPO value number of trainable parameters: {num_trainable:,}')
    logger.info(f'PPO value number of frozen parameters: {num_frozen:,}')

    max_train_steps = int(cfg.ppo_update_epochs * (cfg.max_episodes / cfg.ppo_batch_size))
    num_epochs = cfg.max_episodes // cfg.train_rollout_size

    policy_optimizer = create_optimizer(
        model=policy_model,
        lr=cfg.policy_init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
    )

    policy_scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=policy_optimizer,
        init_lr=cfg.policy_init_lr,
        max_lr=cfg.policy_max_lr,
        min_lr=cfg.policy_min_lr,
        warmup_steps=cfg.policy_lr_warmup_steps,
        max_decay_steps=max_train_steps,
    )

    value_optimizer = create_optimizer(
        model=value_model,
        lr=cfg.value_init_lr,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adam_betas,
        fused=cfg.adam_fused,
    )

    value_scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=value_optimizer,
        init_lr=cfg.value_init_lr,
        max_lr=cfg.value_max_lr,
        min_lr=cfg.value_min_lr,
        warmup_steps=cfg.value_lr_warmup_steps,
        max_decay_steps=max_train_steps,
    )

    ppo_agent = PPOAgent(
        tokenizer=tokenizer,
        policy_model=policy_model,
        policy_optimizer=policy_optimizer,
        policy_scheduler=policy_scheduler,
        value_model=value_model,
        value_optimizer=value_optimizer,
        value_scheduler=value_scheduler,
        reward_model=reward_model,
        ref_model=ref_model,
        ppo_update_epochs=cfg.ppo_update_epochs,
        ppo_batch_size=cfg.ppo_batch_size,
        policy_micro_bs=cfg.policy_micro_bs,
        value_micro_bs=cfg.value_micro_bs,
        policy_clip_eps=cfg.policy_clip_eps,
        value_clip_eps=cfg.value_clip_eps,
        kl_ctl=AdaptiveKLController(init_kl_coef=cfg.init_kl_coef, adaptive=cfg.adaptive_kl, target=cfg.kl_target, horizon=cfg.adaptive_kl_horizon),
        scale_kl=cfg.scale_kl,
        clip_reward=cfg.clip_reward,
        whiten_rewards=cfg.whiten_rewards,
        whiten_advantages=cfg.whiten_advantages,
        truncate_token=cfg.truncate_token,
        truncate_penalty_value=cfg.truncate_penalty_value,
        entropy_coef=cfg.entropy_coef,
        ptx_coef=cfg.ptx_coef,
        discount=cfg.discount,
        gae_lambda=cfg.gae_lambda,
        policy_grad_clip=cfg.policy_grad_clip,
        value_grad_clip=cfg.value_grad_clip,
        reward_device=cfg.reward_device,
        ref_device=cfg.ref_device,
        policy_device=cfg.policy_device,
        value_device=cfg.value_device,
        ptx_loader=train_ptx_loader,
        tb_writer=SummaryWriter(os.path.join(cfg.log_dir, cfg.policy_model_type)),
    )

    # --------------- Start Training ---------------

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.ckpt_dir, 'policy'), exist_ok=True)
    os.makedirs(os.path.join(cfg.ckpt_dir, 'value'), exist_ok=True)

    logger.info(f'Starting to train the model using RL PPO over {num_epochs} epochs ...')

    for epoch in range(1, num_epochs + 1):  # one epoch is just M episodes
        logger.info(f'Epoch {epoch}')
        logger.info(f'Starting to generate {cfg.train_rollout_size} train episodes ...')

        ppo_agent.run_selfplay(
            epoch=epoch,
            dataset=train_prompt_dataset,
            num_episodes=cfg.train_rollout_size,
            batch_size=cfg.selfplay_batch_size,
            min_gen_len=cfg.min_gen_len,
            max_gen_len=cfg.max_gen_len,
            log_interval=cfg.selfplay_log_interval,
            sample_interval=cfg.selfplay_sample_interval,
            temperature=cfg.train_temperature,
            top_k=cfg.train_top_k,
            top_p=cfg.train_top_p,
            is_training=True,
        )

        assert len(ppo_agent.buffer) >= cfg.train_rollout_size

        logger.info(f'Starting to train the agent using {len(ppo_agent.buffer)} selfplay episodes ...')

        torch.cuda.empty_cache()
        ppo_agent.run_ppo_training_steps(log_interval=cfg.train_log_interval)

        # regular checkpointing
        if epoch % cfg.ckpt_interval == 0:
            create_checkpoint(policy_model, os.path.join(cfg.ckpt_dir, f'policy/{cfg.policy_model_type}-epoch-{epoch}.pth'))
            create_checkpoint(value_model, os.path.join(cfg.ckpt_dir, f'value/{cfg.value_model_type}-epoch-{epoch}.pth'))

        # validation episodes
        if cfg.var_interval > 0 and epoch % cfg.var_interval == 0:
            logger.info(f'Starting to generate {cfg.val_rollout_size} validation episodes ...')
            ppo_agent.run_selfplay(
                epoch=epoch,
                dataset=val_prompt_dataset,
                num_episodes=cfg.val_rollout_size,
                batch_size=cfg.selfplay_batch_size,
                min_gen_len=cfg.min_gen_len,
                max_gen_len=cfg.max_gen_len,
                log_interval=1,
                sample_interval=max(10, cfg.selfplay_sample_interval * 0.1),
                temperature=cfg.val_temperature,
                top_k=cfg.val_top_k,
                top_p=cfg.val_top_p,
                is_training=False,
            )

    # create final checkpoints
    create_checkpoint(policy_model, os.path.join(cfg.ckpt_dir, f'policy/{cfg.policy_model_type}-epoch-{epoch}.pth'))
    create_checkpoint(value_model, os.path.join(cfg.ckpt_dir, f'value/{cfg.value_model_type}-epoch-{epoch}.pth'))


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    main()
