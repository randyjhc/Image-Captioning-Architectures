"""Factory for constructing ImageCaptioningModel from ConfigViT."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_vit.decoder import CaptionDecoder
from model_vit.model import ImageCaptioningModel
from model_vit.vit_encoder import ViTEncoder

if TYPE_CHECKING:
    from train_vit.config import ConfigViT


def build_model_from_config(
    config: ConfigViT,
    vocab_size: int,
    pad_token_id: int,
    pretrained: bool = True,
    freeze: bool | None = None,
) -> ImageCaptioningModel:
    """Construct an ImageCaptioningModel from a ConfigViT.

    Args:
        config: Training/model configuration.
        vocab_size: Number of tokens in the vocabulary.
        pad_token_id: Token id used for padding.
        pretrained: Whether to load pretrained ViT weights. Use True for
                    training, False when weights will be loaded from a checkpoint.
        freeze: Whether to freeze the encoder. If None, uses
                ``config.freeze_encoder``.

    Returns:
        Initialised ImageCaptioningModel (weights not loaded from any checkpoint).
    """
    encoder = ViTEncoder(
        model_name=config.vit_model_name,
        pretrained=pretrained,
        decoder_dim=config.decoder_dim,
        freeze=config.freeze_encoder if freeze is None else freeze,
    )
    decoder = CaptionDecoder(
        vocab_size=vocab_size,
        d_model=config.decoder_dim,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_caption_len,
        pad_token_id=pad_token_id,
    )
    return ImageCaptioningModel(
        encoder=encoder,
        decoder=decoder,
        pad_token_id=pad_token_id,
    )
