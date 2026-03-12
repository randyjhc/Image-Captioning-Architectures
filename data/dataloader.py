"""DataLoader factory and utilities for Flickr8k dataset."""

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader

from .flickr_dataset import FlickrDataset


def _get_collate_fn(collate_fn_type: str, pad_token_id: int = 0) -> Callable:
    """
    Get collate function based on type.

    Args:
        collate_fn_type: Type of collate function ("default" or "padding")
        pad_token_id: Token ID for padding (used if collate_fn_type="padding")

    Returns:
        Collate function

    Raises:
        ValueError: If collate_fn_type is not "default" or "padding"
    """
    if collate_fn_type == "padding":
        return collate_fn_with_padding(pad_token_id=pad_token_id)
    elif collate_fn_type == "default":
        return collate_fn
    else:
        raise ValueError(
            f"Unknown collate_fn_type: {collate_fn_type}. "
            f"Must be 'default' or 'padding'."
        )


def collate_fn(batch: list[tuple[Any, Any]]) -> tuple[torch.Tensor, list[Any]]:
    """
    Collate function for batching image-caption pairs.

    Args:
        batch: List of (image, caption) tuples from dataset

    Returns:
        Tuple of (images, captions):
            - images: Stacked tensor of images [batch_size, C, H, W]
            - captions: List of captions (raw strings or tokenized)

    Note:
        Images are stacked into a tensor, but captions are kept as a list
        because they may have variable lengths. If captions are tokenized,
        a custom collate function should be used for padding.
    """
    # Direct unpacking to avoid intermediate tuple creation
    images = [img for img, _ in batch]
    captions = [cap for _, cap in batch]

    # Stack images into a single tensor
    # Assumes images are already converted to tensors via transform
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)

    return images, captions


def collate_fn_with_padding(
    pad_token_id: int = 0,
) -> Callable[[list[tuple[Any, Any]]], tuple[torch.Tensor, torch.Tensor]]:
    """
    Create a collate function with padding for tokenized captions.

    This is useful when captions are already tokenized and need to be
    padded to the same length for batching.

    Args:
        pad_token_id: Token ID to use for padding

    Returns:
        Collate function that pads captions to max length in batch

    Example:
        >>> collate = collate_fn_with_padding(pad_token_id=0)
        >>> loader = create_dataloader(..., collate_fn=collate)
    """

    def collate(batch: list[tuple[Any, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        # Direct unpacking to avoid intermediate tuple creation
        images = [img for img, _ in batch]
        captions = [cap for _, cap in batch]

        # Stack images
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)

        # Pad captions to max length in batch
        if isinstance(captions[0], (list, torch.Tensor)):
            # Convert to tensors if needed
            caption_tensors = [
                torch.tensor(c) if isinstance(c, list) else c for c in captions
            ]

            # Find max length in batch
            max_len = max(len(c) for c in caption_tensors)
            batch_size = len(caption_tensors)

            # Pre-allocate padded tensor (more efficient than per-caption padding)
            padded = torch.full(
                (batch_size, max_len), pad_token_id, dtype=caption_tensors[0].dtype
            )

            # Fill in-place (avoids creating intermediate tensors)
            for i, c in enumerate(caption_tensors):
                padded[i, : len(c)] = c

            captions = padded

        return images, captions

    return collate


def create_dataloader(
    root_dir: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
    collate_fn_type: str = "default",
    pad_token_id: int = 0,
    max_samples: Optional[int] = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """
    Create a DataLoader for Flickr8k dataset.

    Args:
        root_dir: Root directory containing the Flickr8k dataset
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        transform: Optional transform to apply to images
        tokenizer: Optional tokenizer for caption processing
        collate_fn_type: Type of collate function ("default" or "padding")
        pad_token_id: Token ID for padding (used if collate_fn_type="padding")
        max_samples: Optional limit on number of samples
        **dataset_kwargs: Additional arguments to pass to FlickrDataset

    Returns:
        DataLoader for the Flickr8k dataset

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ...     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        ...                          std=[0.229, 0.224, 0.225])
        ... ])
        >>> loader = create_dataloader(
        ...     root_dir="data/datasets/flickr8k",
        ...     batch_size=32,
        ...     transform=transform
        ... )
    """
    # Create dataset
    dataset = FlickrDataset(
        root_dir=root_dir,
        transform=transform,
        tokenizer=tokenizer,
        max_samples=max_samples,
        **dataset_kwargs,
    )

    # Select collate function
    collate = _get_collate_fn(collate_fn_type, pad_token_id)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return dataloader


def create_split_dataloaders(
    root_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
    collate_fn_type: str = "default",
    pad_token_id: int = 0,
    **dataset_kwargs: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for Flickr8k dataset.

    Args:
        root_dir: Root directory containing the Flickr8k dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        transform: Optional transform to apply to images
        tokenizer: Optional tokenizer for caption processing
        collate_fn_type: Type of collate function ("default" or "padding")
        pad_token_id: Token ID for padding (used if collate_fn_type="padding")
        **dataset_kwargs: Additional arguments to pass to FlickrDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> train_loader, val_loader, test_loader = create_split_dataloaders(
        ...     root_dir="data/datasets/flickr8k",
        ...     batch_size=32,
        ...     transform=transform
        ... )
    """
    # Create split datasets
    train_dataset, val_dataset, test_dataset = FlickrDataset.create_splits(
        root_dir=root_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        transform=transform,
        tokenizer=tokenizer,
        **dataset_kwargs,
    )

    # Select collate function
    collate = _get_collate_fn(collate_fn_type, pad_token_id)

    # Create DataLoaders
    datasets_and_shuffle = [
        (train_dataset, True),
        (val_dataset, False),
        (test_dataset, False),
    ]

    loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        for dataset, shuffle in datasets_and_shuffle
    ]

    return tuple(loaders)
