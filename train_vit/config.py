"""Configuration dataclass for ViT image captioning training and finetuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields


@dataclass
class ConfigViT:
    """
    Centralised configuration for training and finetuning ImageCaptioningModel.

    Environment facts (data_root, device) are passed directly to TrainerViT,
    not stored here, so this config is portable across machines.

    Example:
        >>> config = ConfigViT(freeze_encoder=False, lr=1e-5, num_epochs=5)
        >>> trainer = TrainerViT(config, data_root="data/datasets/flickr8k")
        >>> trainer.fit()
    """

    # ---- Paths ----
    data_root: str = "data/datasets/flickr8k"
    checkpoint_dir: str = "checkpoints/vit"
    image_paths: tuple[str, ...] = (
        "data/datasets/flickr8k/Images/667626_18933d713e.jpg",
        "data/datasets/flickr8k/Images/3637013_c675de7705.jpg",
        "data/datasets/flickr8k/Images/10815824_2997e03d76.jpg",
    )  # Images to caption in generation mode

    # ---- Data ----
    image_size: int = 224
    min_vocab_freq: int = 1
    max_seq_len: int = 34  # CaptionTokenizer fixed output length (covers most Flickr8k)
    max_samples: int | None = None  # None = full dataset; set e.g. 500 for fast runs

    # ---- DataLoader ----
    batch_size: int = 32
    num_workers: int = 0

    # ---- Encoder (ViT) ----
    vit_model_name: str = "vit_base_patch16_224"
    freeze_encoder: bool = True  # Set False for full finetuning

    # ---- Decoder ----
    decoder_dim: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_caption_len: int = 128  # Decoder max_len; separate from tokenizer max_seq_len

    # ---- Optimiser ----
    lr: float = 1e-4
    weight_decay: float = 0.01
    eta_min: float = 0.0  # Minimum LR at end of cosine schedule

    # ---- Training ----
    num_epochs: int = 2
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> ConfigViT:
        """Construct ConfigViT from a dict, ignoring unrecognised keys."""
        logger = logging.getLogger("image_caption")
        known = {f.name for f in fields(cls)}
        unknown = d.keys() - known
        if unknown:
            logger.warning("Unrecognised config keys ignored: %s", sorted(unknown))
        return cls(**{k: v for k, v in d.items() if k in known})
