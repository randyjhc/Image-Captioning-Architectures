import torch
import torch.nn as nn


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
            input_ids: [B, T]
            memory_key_padding_mask: optional [B, N]

        Returns:
            logits: [B, T, vocab_size]
        """
        memory = self.encoder(images)  # [B, N, D]

        logits = self.decoder(
            input_ids=input_ids,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T, vocab_size]

        return logits

    def compute_loss(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        criterion: nn.Module,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing loss.

        Args:
            images: [B, 3, H, W]
            captions: [B, T]
                Full caption sequence including BOS/EOS/PAD
            criterion:
                e.g. nn.CrossEntropyLoss(ignore_index=pad_token_id)

        Returns:
            loss: scalar tensor
            logits: [B, T-1, vocab_size]
        """
        input_ids = captions[:, :-1]  # decoder input
        targets = captions[:, 1:]  # next-token targets

        logits = self.forward(
            images=images,
            input_ids=input_ids,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T-1, vocab_size]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        return loss, logits
