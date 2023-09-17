import sys
import logging


def create_logger(level="INFO"):
    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter(
        fmt="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    veb = logging.INFO
    level = str(level).upper()
    if level == "DEBUG":
        veb = logging.DEBUG
    logger.setLevel(veb)
    logger.addHandler(handler)

    return logger
