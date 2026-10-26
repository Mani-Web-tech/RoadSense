"""
RoadSense Logging Configuration
Structured logging with timestamps, levels, and module tags.
"""
import logging
import sys
from typing import Optional

_LOGGER_INITIALIZED = False

def setup_logger(name: str = "RoadSense", level: int = logging.INFO) -> logging.Logger:
    """
    Initializes and returns a configured logger.
    """
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _LOGGER_INITIALIZED and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        _LOGGER_INITIALIZED = True

    return logger

logger = setup_logger()
