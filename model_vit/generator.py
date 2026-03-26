"""Standalone greedy caption generator for ImageCaptioningModel."""

from __future__ import annotations

from pathlib import Path

import torch

from data.text.vocabulary import CaptionTokenizer

from .model import ImageCaptioningModel


class GeneratorViT:
    """
    Wraps an ImageCaptioningModel to provide greedy caption generation.

    Args:
        model: A trained ImageCaptioningModel.
        tokenizer: CaptionTokenizer used to decode token ids to strings.
                   Provides sos/eos token ids via tokenizer.vocab.
        checkpoint_path: Optional path to a checkpoint file. If provided,
                         loads model weights from ``checkpoint["model_state_dict"]``
                         before generating (mirrors TrainerViT.load_checkpoint,
                         but omits optimizer state).
    """

    def __init__(
        self,
        model: ImageCaptioningModel,
        tokenizer: CaptionTokenizer,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])

    @torch.inference_mode()
    def generate_ids(
        self,
        images: torch.Tensor,
        max_len: int = 30,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Greedy decoding. Returns raw token id tensor.

        Args:
            images: [B, 3, H, W] or [3, H, W]
            max_len: Maximum number of tokens to generate.
            memory_key_padding_mask: Optional [B, N] mask for encoder tokens.

        Returns:
            generated_ids: [B, <= max_len]
        """
        sos_token_id = self.tokenizer.vocab.sos_idx
        eos_token_id = self.tokenizer.vocab.eos_idx

        self.model.eval()

        if images.dim() == 3:
            images = images.unsqueeze(0)

        memory = self.model.encoder(images)  # [B, N, D]
        batch_size = images.size(0)

        generated = torch.full(
            (batch_size, 1),
            fill_value=sos_token_id,
            dtype=torch.long,
            device=images.device,
        )

        for _ in range(max_len - 1):
            logits = self.model.decoder(
                input_ids=generated,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # [B, cur_len, vocab_size]

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            if torch.all(next_token.squeeze(1) == eos_token_id):
                break

        return generated

    def generate_caption(
        self,
        images: torch.Tensor,
        max_len: int = 30,
        skip_special: bool = True,
    ) -> list[str]:
        """
        Generate caption strings for a batch of images.

        Args:
            images: [B, 3, H, W] or [3, H, W]
            max_len: Maximum number of tokens to generate.
            skip_special: If True, omit special tokens (<SOS>, <EOS>, <PAD>) from output.

        Returns:
            List of decoded caption strings, one per image in batch.
        """
        generated = self.generate_ids(images, max_len=max_len)
        return [
            self.tokenizer.vocab.decode(row.tolist(), skip_special=skip_special)
            for row in generated
        ]
