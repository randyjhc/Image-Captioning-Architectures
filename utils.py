"""Shared utilities: logging setup and reproducibility seeding."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch


def logger_setup(name: str = "image_caption") -> logging.Logger:
    """Configure and return a named logger with a StreamHandler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s | %(funcName)s || %(message)s")
        )
        logger.addHandler(handler)
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
