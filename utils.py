"""Shared utilities: logging setup and reproducibility seeding."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch


def logger_setup(
    name: str = "image_caption",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a named logger with a StreamHandler.

    If *log_file* is provided, a FileHandler is also attached so that every
    log record is written to both the terminal and the file simultaneously.
    Parent directories are created automatically.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(levelname)s | %(funcName)s || %(message)s")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    if log_file is not None:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        already_attached = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
            for h in logger.handlers
        )
        if not already_attached:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def seed_everything(seed: int = 42) -> torch.Generator:
    """Seed all RNGs for reproducibility. Returns a seeded torch.Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    return g
