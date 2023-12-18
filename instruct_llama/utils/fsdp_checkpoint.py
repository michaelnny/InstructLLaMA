# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
For saving and loading FSDP model checkpoints
"""
import logging
import torch

import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,  # general optimizer non-sharded, non-flattened params
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from instruct_llama.lora import lora_state_dict_from_full_state_dict

logger = logging.getLogger(__name__)


full_state_model_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
full_state_optim_config = FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=True)


def save_lora_model_checkpoint(model, rank, ckpt_save_path, train_bias: str, train_head, overwrite=False):
    """Save lora weights to checkpoint"""
    save_full_path = Path(ckpt_save_path)
    if rank == 0:
        if not overwrite and save_full_path.exists():
            logger.warning(f'a file with the same name already exists at {save_full_path}, aborting ...')
            return
        else:
            save_dir = save_full_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=full_state_model_config,
        optim_state_dict_config=full_state_optim_config,
    ):
        model_state = model.state_dict()

    logger.debug(f'model state_dict ready on rank {rank}\n')

    if rank == 0:
        logger.debug('--> saving lora model ...')

        lora_state = lora_state_dict_from_full_state_dict(model_state, train_bias, train_head)

        torch.save(lora_state, save_full_path)

        logger.debug(f'--> model checkpoint saved at {save_full_path}\n')


def save_full_state_model_checkpoint(model, rank, ckpt_save_path, overwrite=False):
    """saving model via rank0 cpu streaming and full_state_dict"""

    save_full_path = Path(ckpt_save_path)
    if rank == 0:
        if not overwrite and save_full_path.exists():
            logger.warning(f'a file with the same name already exists at {save_full_path}, aborting ...')
            return
        else:
            save_dir = save_full_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=full_state_model_config,
        optim_state_dict_config=full_state_optim_config,
    ):
        model_state = model.state_dict()

    logger.debug(f'model state_dict ready on rank {rank}\n')

    if rank == 0:
        logger.debug('--> saving model ...')

        torch.save(model_state, save_full_path)

        logger.debug(f'--> model checkpoint saved at {save_full_path}\n')


def load_full_state_model_checkpoint(model, rank, full_state_ckpt_file, strict=True):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at ...
    model_full_state_dict_path = Path(full_state_ckpt_file)
    # is it present ...
    if not model_full_state_dict_path.is_file():
        logger.warning(f'model checkpoint {model_full_state_dict_path} not present. aborting ...')
        return

    model_checkpoint = torch.load(model_full_state_dict_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint, strict=strict)

    logger.debug(f'--> model checkpoint {model_full_state_dict_path} loaded to rank0 cpu')


def save_full_state_optimizer_checkpoint(model, optimizer, rank, ckpt_save_path, overwrite=False):
    """save optimizer state via full state dict"""

    save_full_path = Path(ckpt_save_path)
    if rank == 0:
        if not overwrite and save_full_path.exists():
            logger.warning(f'a file with the same name already exists at {save_full_path}, aborting ...')
            return
        else:
            save_dir = save_full_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f'--> optim state call on rank {rank}\n')

    # pull all sharded optimizer states to rank0 cpu ...
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=full_state_model_config,
        optim_state_dict_config=full_state_optim_config,
    ):
        optim_state = FSDP.optim_state_dict(model, optimizer)

    logger.debug(f'optim state dict ready on {rank} and len of {len(optim_state)}\n')

    if rank == 0:
        logger.debug('--> saving optimizer state ...')

        torch.save(optim_state, save_full_path)

        logger.debug(f'--> optimizer checkpoint saved at {save_full_path}\n')


def load_full_state_optimizer_checkpoint(model, optimizer, rank, full_state_ckpt_file):
    """load an fdsp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """

    # where is the checkpoint at ...
    optim_full_state_dict_path = Path(full_state_ckpt_file)
    # is it present ...
    if not optim_full_state_dict_path.is_file():
        logger.warning(f'optimizer checkpoint {optim_full_state_dict_path} not present. aborting ...')
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optim_full_state_dict_path)

        logger.debug('loaded full optimizer state on rank 0')

    # called from all ranks, though only rank0 has a valid param for full_osd
    _ = FSDP.scatter_full_optim_state_dict(full_osd, model)

    logger.debug(f'optimizer shard loaded on rank {rank}')
