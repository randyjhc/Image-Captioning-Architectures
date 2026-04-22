"""Standalone greedy caption generator for CNN+LSTM ImageCaptioningModel."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from data.text.vocabulary import CaptionTokenizer, Vocabulary
from model_cnn_lstm.cnn_encoder import CNNEncoder
from model_cnn_lstm.lstm_decoder import LSTMDecoder
from model_cnn_lstm.model import ImageCaptioningModel


class GeneratorCNN:
    """
    Wraps a trained CNN+LSTM ImageCaptioningModel to provide greedy caption generation.

    Args:
        model: A trained ImageCaptioningModel.
        tokenizer: CaptionTokenizer used to decode token ids to strings.
        checkpoint_path: Optional path to a checkpoint file. If provided,
                         loads model weights from checkpoint["model_state_dict"].
    """

    def __init__(
        self,
        model: ImageCaptioningModel,
        tokenizer: CaptionTokenizer,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger("image_caption")
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_val_loss = ckpt.get("best_val_loss", "N/A")
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.logger.info(
                f"Checkpoint loaded from {checkpoint_path}, best_val_loss={best_val_loss:.4f}"
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        data_root: str | Path,
    ) -> GeneratorCNN:
        """Build a GeneratorCNN from a checkpoint, without a Trainer.

        Loads the saved ConfigCNN from the checkpoint to reconstruct the model
        architecture, rebuilds the vocabulary from vocab_word2idx, and loads the
        model weights.

        Args:
            checkpoint_path: Path to a checkpoint saved by TrainerCNN.
            data_root: Path to the dataset root (not used if vocab is in checkpoint).

        Returns:
            GeneratorCNN instance with weights loaded.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt["config"]  # ConfigCNN

        # Rebuild vocabulary from checkpoint
        vocab = Vocabulary()
        vocab.word2idx = ckpt["vocab_word2idx"]
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        tokenizer = CaptionTokenizer(vocab, max_seq_len=config.max_seq_len)

        # Rebuild model from config
        encoder = CNNEncoder(
            embed_size=config.embed_size,
            pretrained=False,
            freeze=False,
        )
        decoder = LSTMDecoder(
            vocab_size=len(vocab),
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            pad_token_id=vocab.pad_idx,
        )
        model = ImageCaptioningModel(encoder, decoder, pad_token_id=vocab.pad_idx)

        return cls(model, tokenizer, checkpoint_path=checkpoint_path)

    @torch.inference_mode()
    def generate_caption(
        self,
        images: torch.Tensor,
        max_len: int = 30,
        skip_special: bool = True,
    ) -> list[str]:
        """Generate caption strings for a batch of images.

        Args:
            images: [B, 3, H, W] or [3, H, W]
            max_len: Maximum number of tokens to generate.
            skip_special: If True, omit special tokens from output.

        Returns:
            List of decoded caption strings, one per image in batch.
        """
        self.model.eval()
        if images.dim() == 3:
            images = images.unsqueeze(0)

        generated = self.model.generate(
            images,
            sos_token_id=self.tokenizer.vocab.sos_idx,
            eos_token_id=self.tokenizer.vocab.eos_idx,
            max_len=max_len,
        )
        return [
            self.tokenizer.vocab.decode(row.tolist(), skip_special=skip_special)
            for row in generated
        ]
