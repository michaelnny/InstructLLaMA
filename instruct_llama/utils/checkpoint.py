# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


import os
import json
import torch
import logging
from typing import Optional, Tuple

from instruct_llama.models.model import Transformer
from instruct_llama.models.lora import lora_state_dict
from instruct_llama.core.rl_ppo import PPOAgent
from instruct_llama.utils.fsdp_checkpoint import save_full_state_model_checkpoint
from instruct_llama.utils.normalizer import BaseNormalizer


logger = logging.getLogger(__name__)


def _check_file_path(file_path) -> None:
    if not file_path or file_path == '':
        raise ValueError(f'Invalid checkpoint file path {file_path!r}')
    if os.path.exists(file_path):
        logger.warning(f'Existing checkpoint file at {file_path!r} will be overwritten')


def _save_model_meta(model: Transformer, save_dir: str) -> None:
    meta_file = os.path.join(save_dir, 'params.json')
    if not os.path.exists(meta_file):
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(model.params.dict(), f, indent=2)
        logger.info(f'Model meta params saved at {meta_file!r}')


def create_checkpoint(model: Transformer, full_path: str) -> None:
    _check_file_path(full_path)
    ckpt_state = model.state_dict()
    torch.save(ckpt_state, full_path)
    logger.info(f'Model checkpoint saved at {full_path!r}')

    _save_model_meta(model, os.path.dirname(full_path))


def create_lora_checkpoint(model: Transformer, full_path: str, train_bias: bool, additional_layers: Optional[Tuple[str]] = None) -> None:
    _check_file_path(full_path)

    ckpt_state = lora_state_dict(model, train_bias=train_bias, additional_layers=additional_layers)
    torch.save(ckpt_state, full_path)
    logger.info(f'LoRA model checkpoint saved at {full_path!r}')

    # save LoRA configuration params so we can re-create the same model after training, which is required to merge weights
    _save_model_meta(model, os.path.dirname(full_path))


def create_fsdp_full_checkpoint(model: Transformer, rank: int, full_path: str) -> None:
    _check_file_path(full_path)

    save_full_state_model_checkpoint(model, rank, full_path, True)
    logger.info(f'FSDP model full state checkpoint saved at {full_path!r}')

    # save model configuration params
    _save_model_meta(model, os.path.dirname(full_path))


def create_normalizer_checkpoint(norm: BaseNormalizer, full_path: str) -> None:
    _check_file_path(full_path)
    torch.save(norm.state_dict(), full_path)
    logger.info(f'Normalizer checkpoint saved at {full_path!r}')


def create_agent_checkpoint(agent: PPOAgent, full_path: str) -> None:
    _check_file_path(full_path)
    torch.save(agent.state_dict(), full_path)
    logger.info(f'Agent checkpoint saved at {full_path!r}')
