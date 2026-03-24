import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id

        # Token embedding for caption input ids
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )

        # Learnable positional embedding
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Important: shapes are [B, T, D]
        )

        # Stack of decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        # Final language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a causal mask so position i cannot attend to positions > i.
        Shape: [T, T]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()
        return mask

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        returns:   [B, T]
        """
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
            Caption token ids used as decoder input (usually captions[:, :-1])

        memory: [B, N, D]
            Visual features from encoder (ViT patch tokens after projection)

        memory_key_padding_mask: [B, N] or None
            Optional mask for encoder tokens. Usually None for ViT patch tokens.

        Returns:
            logits: [B, T, vocab_size]
        """

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len={self.max_len}"
            )

        # ---- Embeddings ----
        token_embeds = self.token_embedding(input_ids)  # [B, T, D]
        position_ids = self._get_position_ids(input_ids)
        position_embeds = self.position_embedding(position_ids)  # [B, T, D]

        tgt = token_embeds + position_embeds
        tgt = self.dropout(tgt)

        # ---- Masks ----
        tgt_mask = self._generate_causal_mask(seq_len, input_ids.device)  # [T, T]

        tgt_key_padding_mask = input_ids == self.pad_token_id  # [B, T]

        # ---- Transformer Decoder ----
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [B, T, D]

        # ---- Output logits ----
        logits = self.lm_head(decoded)  # [B, T, vocab_size]

        return logits
