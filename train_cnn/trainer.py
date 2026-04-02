"""Trainer for CNN+LSTM image captioning."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data import create_split_dataloaders_with_vocab
from data.image.transforms import get_train_transforms, get_val_transforms
from data.text.vocabulary import CaptionTokenizer, Vocabulary
from model_cnn_lstm import CNNEncoder, ImageCaptioningModel, LSTMDecoder
from utils import seed_everything

from .config import ConfigCNN


class TrainerCNN:
    """Self-contained trainer for the CNN+LSTM baseline."""

    def __init__(
        self,
        config: ConfigCNN,
        data_root: str | Path,
        device: torch.device | str | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root)
        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._logger = logging.getLogger("image_caption")

        (
            self._train_loader,
            self._val_loader,
            self._test_loader,
            self._vocab,
            self._tokenizer,
        ) = self._build_dataloaders(seed_everything(config.seed))
        self._train_dataset = self._train_loader.dataset
        self._val_dataset = self._val_loader.dataset
        self._test_dataset = self._test_loader.dataset
        self._model = self._build_model().to(self._device)
        self._criterion = nn.CrossEntropyLoss(ignore_index=self._vocab.pad_idx)
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=config.lr_reduce_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.eta_min,
        )
        self._logger.info("=" * 20)
        self._logger.info("Training Setup")
        self._logger.info("=" * 20)
        self._logger.info(f"Device={self._device}")
        self._logger.info(
            "batch_size=%s | lr=%s | epochs=%s | vocab_size=%s",
            self.config.batch_size,
            self.config.lr,
            self.config.num_epochs,
            len(self._vocab),
        )

    def fit(self, num_epochs: int | None = None) -> None:
        """Run the train + validation loop and save best/latest checkpoints."""
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        best_val_loss = float("inf")
        wait = 0

        if self._checkpoint_dir is not None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            lr = self._optimizer.param_groups[0]["lr"]

            self._logger.info(
                f"Epoch={epoch + 1}/{epochs} | "
                f"lr={lr:.6f} | train={train_loss:.4f} | val={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                if self._checkpoint_dir is not None:
                    self.save_checkpoint(
                        self._checkpoint_dir / "best.pt",
                        epoch + 1,
                        best_val_loss=best_val_loss,
                    )
                    self._logger.info(
                        f"Saved best checkpoint to {self._checkpoint_dir / 'best.pt'}"
                    )
            else:
                wait += 1
                if wait >= self.config.patience:
                    self._logger.info(
                        f"Early stopping at epoch {epoch + 1} | best_val={best_val_loss:.4f}"
                    )
                    break

            self._scheduler.step(val_loss)

    def evaluate(self) -> float:
        """Run one validation pass. Returns average validation loss."""
        return self._validate_one_epoch()

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        best_val_loss: float | None = None,
    ) -> None:
        """Save model weights, optimizer state, config, and vocab."""
        torch.save(
            {
                "model_state_dict": copy.deepcopy(self._model.state_dict()),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict(),
                "config": self.config,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "vocab_word2idx": self._vocab.word2idx,
            },
            path,
        )
        self._logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path) -> dict[str, object]:
        """Load model, optimizer, scheduler, and vocabulary state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)

        if "vocab_word2idx" in checkpoint:
            vocab = Vocabulary()
            vocab.word2idx = checkpoint["vocab_word2idx"]
            vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
            self._vocab = vocab
            self._tokenizer = CaptionTokenizer(
                vocab, max_seq_len=self.config.max_seq_len
            )
            self._train_dataset.tokenizer = self._tokenizer
            self._val_dataset.tokenizer = self._tokenizer
            self._test_dataset.tokenizer = self._tokenizer

        self._model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._logger.info(f"Checkpoint loaded from {path}")

        return {
            "epoch": checkpoint.get("epoch"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        }

    @property
    def model(self) -> ImageCaptioningModel:
        return self._model

    @property
    def tokenizer(self) -> CaptionTokenizer:
        return self._tokenizer

    @property
    def val_dataset(self):
        return self._val_dataset

    def _build_dataloaders(
        self,
        generator: torch.Generator,
    ):
        return create_split_dataloaders_with_vocab(
            root_dir=self.data_root,
            min_vocab_freq=self.config.min_vocab_freq,
            max_seq_len=self.config.max_seq_len,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            generator=generator,
            train_transform=get_train_transforms(self.config.image_size),
            val_transform=get_val_transforms(self.config.image_size),
            test_transform=get_val_transforms(self.config.image_size),
            max_samples=self.config.max_samples,
        )

    def _build_model(self) -> ImageCaptioningModel:
        encoder = CNNEncoder(
            embed_size=self.config.embed_size,
            pretrained=self.config.pretrained_encoder,
            freeze=self.config.freeze_encoder,
        )
        decoder = LSTMDecoder(
            vocab_size=len(self._vocab),
            embed_size=self.config.embed_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            pad_token_id=self._vocab.pad_idx,
        )
        return ImageCaptioningModel(
            encoder=encoder,
            decoder=decoder,
            pad_token_id=self._vocab.pad_idx,
        )

    def _train_one_epoch(self) -> float:
        self._model.train()
        total_loss = torch.zeros(1, device=self._device)

        for images, captions in tqdm(self._train_loader, desc="Train", unit="batch"):
            images = images.to(self._device, non_blocking=True)
            captions = captions.to(self._device, non_blocking=True)

            self._optimizer.zero_grad()
            loss, _ = self._model.compute_loss(images, captions, self._criterion)
            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self.config.grad_clip
                )
            self._optimizer.step()
            total_loss += loss.detach()

        return (total_loss / len(self._train_loader)).item()

    @torch.no_grad()
    def _validate_one_epoch(self) -> float:
        self._model.eval()
        total_loss = torch.zeros(1, device=self._device)

        for images, captions in tqdm(self._val_loader, desc="Eval", unit="batch"):
            images = images.to(self._device, non_blocking=True)
            captions = captions.to(self._device, non_blocking=True)

            loss, _ = self._model.compute_loss(images, captions, self._criterion)
            total_loss += loss.detach()

        return (total_loss / len(self._val_loader)).item()
