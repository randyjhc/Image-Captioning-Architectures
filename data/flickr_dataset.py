"""Flickr dataset with lazy loading for images and text interface."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from torch.utils.data import Dataset

from .image.image_utils import load_image, validate_image_path

if TYPE_CHECKING:
    from PIL import Image


class FlickrDataset(Dataset):
    """
    Flickr dataset with lazy image loading and text processing interface.

    Supports Flickr8k and Flickr30k (and similar datasets) via configurable
    image directory name and CSV column names.

    Args:
        root_dir: Root directory containing the dataset
        captions_file: Name of the captions file (default: "captions.txt")
        image_dir_name: Subdirectory under root_dir containing images
            (default: "Images" for Flickr8k; use "flickr30k_images" for Flickr30k)
        image_col: CSV column name for image filenames. If None, auto-detected
            from the header using the priority list: image_name, image, filename, file.
        caption_col: CSV column name for captions. If None, auto-detected
            from the header using the priority list: comment, caption, text, description.
        transform: Optional transform to apply to images
        tokenizer: Optional tokenizer for caption processing (interface only)
        max_samples: Optional limit on number of samples (for debugging)

    Example — Flickr8k:
        >>> dataset = FlickrDataset(root_dir="data/datasets/flickr8k")

    Example — Flickr30k:
        >>> dataset = FlickrDataset(
        ...     root_dir="data/datasets/flickr30k",
        ...     captions_file="captions.txt",
        ...     image_dir_name="flickr30k_images",
        ... )
    """

    _IMAGE_COL_CANDIDATES: tuple[str, ...] = ("image_name", "image")
    _CAPTION_COL_CANDIDATES: tuple[str, ...] = ("comment", "caption")

    def __init__(
        self,
        root_dir: str | Path,
        captions_file: str = "captions.txt",
        image_dir_name: str = "Images",
        image_col: Optional[str] = None,
        caption_col: Optional[str] = None,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_samples: Optional[int] = None,
    ):
        """Initialize FlickrDataset."""
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir_name
        self._image_col = image_col  # None = auto-detect from CSV header
        self._caption_col = caption_col
        self.transform = transform
        self.tokenizer = tokenizer

        # Validate paths
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")

        captions_path = self.root_dir / captions_file
        if not captions_path.exists():
            raise ValueError(f"Captions file does not exist: {captions_path}")

        # Load captions and build samples list
        # Each sample is (image_filename, caption_text)
        self.samples: list[tuple[str, str]] = []
        self._load_captions(captions_path, max_samples)

    def _load_captions(
        self, captions_path: Path, max_samples: Optional[int] = None
    ) -> None:
        """
        Load captions from a CSV file.

        Supports comma-separated (Flickr8k default) and pipe-separated
        (Flickr30k results.csv) files. The delimiter is auto-detected from
        the header line. Column names are also auto-detected if not specified.
        """
        with open(captions_path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            delimiter = "|" if "|" in header_line else ","
            fieldnames = [c.strip() for c in header_line.split(delimiter)]
            reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=delimiter)

            image_col = self._image_col or next(
                (c for c in self._IMAGE_COL_CANDIDATES if c in fieldnames), None
            )
            if image_col is None:
                raise ValueError(
                    f"Cannot auto-detect image column in {captions_path}. "
                    f"Columns found: {fieldnames}. "
                    f"Set image_col explicitly."
                )

            caption_col = self._caption_col or next(
                (c for c in self._CAPTION_COL_CANDIDATES if c in fieldnames), None
            )
            if caption_col is None:
                raise ValueError(
                    f"Cannot auto-detect caption column in {captions_path}. "
                    f"Columns found: {fieldnames}. "
                    f"Set caption_col explicitly."
                )

            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                image_filename = row[image_col].strip()
                caption = row[caption_col].strip()
                self.samples.append((image_filename, caption))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image | Any, str | Any]:
        """
        Load and return image and caption at the given index.

        Args:
            idx: Index of the sample to load

        Returns:
            Tuple of (image, caption):
                - image: Processed image (PIL Image or Tensor)
                - caption: Raw caption string or tokenized caption

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image loading fails
        """
        image_filename, caption = self.samples[idx]

        # Lazy loading: image is loaded here, not in __init__
        image_path = validate_image_path(self.image_dir, image_filename)
        image = load_image(image_path, convert_mode="RGB")

        # Apply image transform if provided
        if self.transform is not None:
            image = self.transform(image)

        if self.tokenizer is not None:
            caption = self.tokenizer.encode(caption)

        return image, caption

    def get_image_filename(self, idx: int) -> str:
        """Get the image filename for a given index."""
        return self.samples[idx][0]

    def get_caption(self, idx: int) -> str:
        """Get the raw caption text for a given index."""
        return self.samples[idx][1]

    @staticmethod
    def create_splits(
        root_dir: str | Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        **kwargs: Any,
    ) -> tuple[FlickrDataset, FlickrDataset, FlickrDataset]:
        """
        Create train/val/test splits of the dataset.

        Args:
            root_dir: Root directory containing the Flickr8k dataset
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducibility
            train_transform: Optional transform for training split (overrides kwargs transform)
            val_transform: Optional transform for validation split (overrides kwargs transform)
            test_transform: Optional transform for test split (overrides kwargs transform)
            **kwargs: Additional arguments to pass to FlickrDataset
                Note: If 'transform' is in kwargs and split-specific transforms are None,
                      the kwargs transform will be used for that split.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)

        Example:
            >>> from data.image.transforms import get_train_transforms, get_val_transforms
            >>> train_ds, val_ds, test_ds = FlickrDataset.create_splits(
            ...     root_dir="data/datasets/flickr8k",
            ...     train_transform=get_train_transforms(224),  # With augmentation
            ...     val_transform=get_val_transforms(224),      # Without augmentation
            ...     test_transform=get_val_transforms(224),     # Without augmentation
            ... )
        """
        import random

        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Load full dataset once to get all samples
        full_dataset = FlickrDataset(root_dir, **kwargs)

        # Get unique image filenames (each has 5 captions)
        unique_images = list(dict.fromkeys(img for img, _ in full_dataset.samples))
        random.seed(seed)
        random.shuffle(unique_images)

        # Split images
        n_images = len(unique_images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        train_images = set(unique_images[:n_train])
        val_images = set(unique_images[n_train : n_train + n_val])
        test_images = set(unique_images[n_train + n_val :])

        # Create datasets and assign filtered samples (no redundant I/O)
        split_configs = [
            (train_images, train_transform),
            (val_images, val_transform),
            (test_images, test_transform),
        ]

        datasets = []
        for split_images, split_transform in split_configs:
            dataset = FlickrDataset.__new__(FlickrDataset)
            dataset.root_dir = full_dataset.root_dir
            dataset.image_dir = full_dataset.image_dir
            dataset.transform = (
                split_transform
                if split_transform is not None
                else full_dataset.transform
            )
            dataset.tokenizer = full_dataset.tokenizer
            dataset.samples = [s for s in full_dataset.samples if s[0] in split_images]
            datasets.append(dataset)

        train_dataset, val_dataset, test_dataset = datasets

        return train_dataset, val_dataset, test_dataset
