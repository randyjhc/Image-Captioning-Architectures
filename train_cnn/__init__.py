"""Training utilities for the CNN+LSTM image captioning baseline."""

from .config import ConfigCNN
from .trainer import TrainerCNN

__all__ = ["ConfigCNN", "TrainerCNN"]
