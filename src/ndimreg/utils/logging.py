from __future__ import annotations

from contextlib import contextmanager

from loguru import logger


@contextmanager
def disable_loguru():
    """TODO."""
    logger.disable("")  # Disable logging for the entire app
    try:
        yield
    finally:
        logger.enable("")  # Re-enable logging
