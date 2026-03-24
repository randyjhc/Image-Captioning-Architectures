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

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        sos_token_id: int,
        eos_token_id: int,
        max_len: int = 30,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Greedy decoding.

        Args:
            images: [B, 3, H, W] or [3, H, W]
            sos_token_id: start token id
            eos_token_id: end token id
            max_len: max generated length

        Returns:
            generated_ids: [B, <= max_len]
        """
        self.eval()

        if images.dim() == 3:
            images = images.unsqueeze(0)

        memory = self.encoder(images)  # [B, N, D]
        batch_size = images.size(0)

        generated = torch.full(
            (batch_size, 1),
            fill_value=sos_token_id,
            dtype=torch.long,
            device=images.device,
        )

        for _ in range(max_len - 1):
            logits = self.decoder(
                input_ids=generated,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # [B, cur_len, vocab_size]

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            if torch.all(next_token.squeeze(1) == eos_token_id):
                break

        return generated
