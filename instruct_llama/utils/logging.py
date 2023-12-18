# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


import sys
import logging


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
