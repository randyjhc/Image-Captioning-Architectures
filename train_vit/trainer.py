"""TrainerViT: encapsulates the full training loop for ImageCaptioningModel."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.flickr_dataset import FlickrDataset
from data.image.transforms import get_train_transforms, get_val_transforms
from data.text.vocabulary import CaptionTokenizer, Vocabulary
from model_vit.decoder import CaptionDecoder
from model_vit.model import ImageCaptioningModel
from model_vit.vit_encoder import ViTEncoder

from .config import ConfigViT


class TrainerViT:
    """
    Self-contained trainer for ImageCaptioningModel.

    Builds vocab, tokenizer, dataloaders, model, optimiser, and criterion
    from a ConfigViT and a data root path. Call fit() to run the training loop.

    Args:
        config: Hyperparameter configuration.
        data_root: Path to flickr8k dataset root (contains captions.txt and Images/).
        device: Torch device. Defaults to CUDA if available, else CPU.
        checkpoint_dir: Directory to save checkpoints. None disables checkpointing.

    Example:
        >>> config = ConfigViT(freeze_encoder=False, lr=1e-5, num_epochs=5)
        >>> trainer = TrainerViT(config, data_root="data/datasets/flickr8k")
        >>> trainer.fit()
    """

    def __init__(
        self,
        config: ConfigViT,
        data_root: str | Path,
        device: torch.device | str | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self._vocab, self._tokenizer = self._build_vocab_and_tokenizer()
        self._train_loader, self._val_loader, self._test_loader, self._val_ds = (
            self._build_dataloaders()
        )
        self._model = self._build_model().to(self._device)
        self._criterion = nn.CrossEntropyLoss(ignore_index=self._vocab.pad_idx)
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int | None = None) -> None:
        """
        Run the train + validate loop.

        Args:
            num_epochs: Override config.num_epochs for this run.
        """
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs

        if self._checkpoint_dir is not None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()

            print(
                f"epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

            if self._checkpoint_dir is not None:
                self.save_checkpoint(self._checkpoint_dir / "latest.pt")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(self._checkpoint_dir / "best.pt")

    def evaluate(self) -> float:
        """Run one validation pass. Returns average val loss."""
        return self._validate_one_epoch()

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights, optimiser state, and config to a file."""
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights and optimiser state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> ImageCaptioningModel:
        return self._model

    @property
    def tokenizer(self) -> CaptionTokenizer:
        return self._tokenizer

    @property
    def val_dataset(self) -> FlickrDataset:
        return self._val_ds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_vocab_and_tokenizer(self) -> tuple[Vocabulary, CaptionTokenizer]:
        captions_file = self.data_root / "captions.txt"
        with open(captions_file, encoding="utf-8") as f:
            all_captions = [row["caption"] for row in csv.DictReader(f)]

        vocab = Vocabulary.build_from_captions(
            all_captions, min_freq=self.config.min_vocab_freq
        )
        tokenizer = CaptionTokenizer(vocab, max_seq_len=self.config.max_seq_len)
        return vocab, tokenizer

    def _build_dataloaders(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader, FlickrDataset]:
        train_ds, val_ds, test_ds = FlickrDataset.create_splits(
            root_dir=self.data_root,
            train_transform=get_train_transforms(self.config.image_size),
            val_transform=get_val_transforms(self.config.image_size),
            test_transform=get_val_transforms(self.config.image_size),
            tokenizer=self._tokenizer,
        )

        pin_memory = self._device.type == "cuda"
        persistent = self.config.num_workers > 0

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
        return train_loader, val_loader, test_loader, val_ds

    def _build_model(self) -> ImageCaptioningModel:
        encoder = ViTEncoder(
            model_name=self.config.vit_model_name,
            pretrained=True,
            decoder_dim=self.config.decoder_dim,
            freeze=self.config.freeze_encoder,
        )
        decoder = CaptionDecoder(
            vocab_size=len(self._vocab),
            d_model=self.config.decoder_dim,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_len=self.config.max_caption_len,
            pad_token_id=self._vocab.pad_idx,
        )
        return ImageCaptioningModel(
            encoder=encoder,
            decoder=decoder,
            pad_token_id=self._vocab.pad_idx,
        )

    def _train_one_epoch(self) -> float:
        self._model.train()
        total_loss = 0.0

        for batch_idx, (images, captions) in enumerate(self._train_loader):
            images = images.to(self._device)
            captions = captions.to(self._device)

            self._optimizer.zero_grad()
            loss, _logits = self._model.compute_loss(images, captions, self._criterion)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  batch={batch_idx} loss={loss.item():.4f}")

        return total_loss / len(self._train_loader)

    @torch.no_grad()
    def _validate_one_epoch(self) -> float:
        self._model.eval()
        total_loss = 0.0

        for images, captions in self._val_loader:
            images = images.to(self._device)
            captions = captions.to(self._device)

            loss, _logits = self._model.compute_loss(images, captions, self._criterion)
            total_loss += loss.item()

        return total_loss / len(self._val_loader)
