# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Dict
import sys
import logging

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class DummyLogger:
    def __init__(self):
        pass

    def _noop(self, *args, **kwargs):
        pass

    info = warning = debug = _noop


def create_logger(level='INFO', rank=0):
    if rank == 0:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        veb = logging.INFO
        level = str(level).upper()
        if level == 'DEBUG':
            veb = logging.DEBUG
        logger.setLevel(veb)
        logger.addHandler(handler)

        return logger
    else:
        return DummyLogger()


def log_statistics(tb_writer: SummaryWriter, train_steps: int, stats: Dict, is_training: bool) -> None:
    logger.info(f'Training steps {train_steps}, is status for validation: {not is_training}')
    logger.info(stats)

    if tb_writer is not None:
        tb_tag = 'train' if is_training else 'val'
        for k, v in stats.items():
            tb_writer.add_scalar(f'{tb_tag}/{k}', v, train_steps)
