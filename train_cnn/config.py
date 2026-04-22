"""Configuration dataclass for CNN+LSTM image captioning training."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields


@dataclass
class ConfigCNN:
    """Portable configuration for CNN+LSTM training."""

    # ---- Paths ----
    data_root: str = "data/datasets/flickr8k"
    checkpoint_dir: str = "checkpoints/cnn_lstm"
    log_file: str | None = None
    image_paths: tuple[str, ...] = (
        "data/datasets/flickr8k/Images/667626_18933d713e.jpg",
        "data/datasets/flickr8k/Images/3637013_c675de7705.jpg",
        "data/datasets/flickr8k/Images/10815824_2997e03d76.jpg",
    )

    # ---- Data ----
    image_size: int = 224
    image_dir_name: str = "Images"   # [ADDED] image subdir; "Images" works for both Flickr8k and Flickr30k (if downloaded via script); set "flickr30k_images" for manual download
    image_col: str | None = None     # [ADDED] None = auto-detect from CSV header
    caption_col: str | None = None   # [ADDED] None = auto-detect from CSV header
    min_vocab_freq: int = 1
    max_seq_len: int = 34
    max_samples: int | None = None

    # ---- DataLoader ----
    batch_size: int = 32
    num_workers: int = 0

    # ---- Model ----
    embed_size: int = 512
    hidden_size: int = 512
    num_layers: int = 1
    pretrained_encoder: bool = True
    freeze_encoder: bool = True

    # ---- Optimiser ----
    lr: float = 1e-4
    weight_decay: float = 0.01
    eta_min: float = 0.0
    lr_reduce_factor: float = 0.5
    lr_scheduler_patience: int = 2

    # ---- Training ----
    num_epochs: int = 10
    seed: int = 42
    grad_clip: float = 1.0
    patience: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigCNN":
        """Construct ConfigCNN from a dict, ignoring unrecognised keys."""
        logger = logging.getLogger("image_caption")
        known = {f.name for f in fields(cls)}
        unknown = d.keys() - known
        if unknown:
            logger.warning("Unrecognised config keys ignored: %s", sorted(unknown))
        return cls(**{k: v for k, v in d.items() if k in known})
