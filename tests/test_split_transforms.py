"""Test different transforms for train/val/test splits."""

from pathlib import Path

from data import FlickrDataset
from data.image.transforms import get_train_transforms, get_val_transforms


def test_split_specific_transforms():
    """Test creating splits with different transforms per split."""
    dataset_path = Path("data/datasets/flickr8k")

    # Create a shared transform for val and test to verify they can be the same
    val_test_transform = get_val_transforms(224)

    # Create splits with different transforms
    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=dataset_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        train_transform=get_train_transforms(224),  # With augmentation
        val_transform=val_test_transform,  # Without augmentation
        test_transform=val_test_transform,  # Same instance as val
        max_samples=100,
    )

    # Verify dataset sizes
    assert len(train_ds) > 0, "Train dataset should not be empty"
    assert len(val_ds) > 0, "Val dataset should not be empty"
    assert len(test_ds) > 0, "Test dataset should not be empty"

    # Verify approximate split ratios (with max_samples=100)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total <= 100, f"Total samples {total} should not exceed max_samples=100"

    # Get samples from each split
    train_img, train_caption = train_ds[0]
    val_img, val_caption = val_ds[0]
    test_img, test_caption = test_ds[0]

    # Verify image shapes are correct (C, H, W) format
    assert train_img.shape == (
        3,
        224,
        224,
    ), f"Train image shape should be (3, 224, 224), got {train_img.shape}"
    assert val_img.shape == (
        3,
        224,
        224,
    ), f"Val image shape should be (3, 224, 224), got {val_img.shape}"
    assert test_img.shape == (
        3,
        224,
        224,
    ), f"Test image shape should be (3, 224, 224), got {test_img.shape}"

    # Verify captions exist
    assert train_caption is not None, "Train caption should not be None"
    assert val_caption is not None, "Val caption should not be None"
    assert test_caption is not None, "Test caption should not be None"

    # Verify transforms are correctly assigned
    assert (
        train_ds.transform is not val_ds.transform
    ), "Train transform should differ from val transform (augmentation vs no augmentation)"
    assert (
        val_ds.transform is test_ds.transform
    ), "Val and test should share the same transform instance"
    assert (
        val_ds.transform is val_test_transform
    ), "Val should use the provided transform instance"


def test_default_transform_fallback():
    """Test that splits use kwargs transform if split-specific not provided."""
    dataset_path = Path("data/datasets/flickr8k")

    # Create splits with only kwargs transform (old behavior)
    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=dataset_path,
        transform=get_val_transforms(224),  # All splits use this
        max_samples=50,
    )

    # Verify all splits use the same transform instance
    assert (
        train_ds.transform is val_ds.transform
    ), "Train and val should use the same transform instance when only kwargs transform is provided"
    assert (
        val_ds.transform is test_ds.transform
    ), "Val and test should use the same transform instance when only kwargs transform is provided"

    # Verify datasets are created
    assert len(train_ds) > 0, "Train dataset should not be empty"
    assert len(val_ds) > 0, "Val dataset should not be empty"
    assert len(test_ds) > 0, "Test dataset should not be empty"


def test_mixed_transforms():
    """Test providing some split-specific and some default."""
    dataset_path = Path("data/datasets/flickr8k")

    # Only specify train_transform, others use kwargs
    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=dataset_path,
        transform=get_val_transforms(224),  # Default for val/test
        train_transform=get_train_transforms(224),  # Override for train
        max_samples=50,
    )

    # Verify train uses specific transform while val/test use default
    assert (
        train_ds.transform is not val_ds.transform
    ), "Train should use specific transform, different from val"
    assert (
        val_ds.transform is test_ds.transform
    ), "Val and test should use the same default transform instance"

    # Verify datasets are created
    assert len(train_ds) > 0, "Train dataset should not be empty"
    assert len(val_ds) > 0, "Val dataset should not be empty"
    assert len(test_ds) > 0, "Test dataset should not be empty"

    # Verify we can load samples
    train_img, _ = train_ds[0]
    val_img, _ = val_ds[0]

    assert train_img.shape == (
        3,
        224,
        224,
    ), f"Train image shape should be (3, 224, 224), got {train_img.shape}"
    assert val_img.shape == (
        3,
        224,
        224,
    ), f"Val image shape should be (3, 224, 224), got {val_img.shape}"
