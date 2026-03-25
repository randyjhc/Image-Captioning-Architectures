"""Configuration dataclass for ViT image captioning training and finetuning."""

from __future__ import annotations

from dataclasses import dataclass


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

    # ---- Data ----
    image_size: int = 224
    min_vocab_freq: int = 1
    max_seq_len: int = 34  # CaptionTokenizer fixed output length (covers most Flickr8k)

    # ---- DataLoader ----
    batch_size: int = 8
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

    # ---- Training ----
    num_epochs: int = 2
