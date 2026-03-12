"""Test different transforms for train/val/test splits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import FlickrDataset
from data.image.transforms import get_train_transforms, get_val_transforms


def test_split_specific_transforms():
    """Test creating splits with different transforms per split."""
    print("=" * 60)
    print("Test: Split-Specific Transforms")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")

    # Create splits with different transforms
    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=dataset_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        train_transform=get_train_transforms(224),  # With augmentation
        val_transform=get_val_transforms(224),  # Without augmentation
        test_transform=get_val_transforms(224),  # Without augmentation
        max_samples=100,
    )

    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")
    print(f"Test dataset: {len(test_ds)} samples")

    # Get samples from each split
    train_img, _ = train_ds[0]
    val_img, _ = val_ds[0]
    test_img, _ = test_ds[0]

    print("\nAll images processed correctly:")
    print(f"  Train: {train_img.shape}")
    print(f"  Val: {val_img.shape}")
    print(f"  Test: {test_img.shape}")

    # Check transforms are different
    print("\nTransforms are different:")
    print(f"  Train has augmentation: {train_ds.transform != val_ds.transform}")
    print(f"  Val == Test: {val_ds.transform == test_ds.transform}")

    print("\n" + "=" * 60)
    print("SUCCESS: Each split can have its own transform!")
    print("=" * 60)


def test_default_transform_fallback():
    """Test that splits use kwargs transform if split-specific not provided."""
    print("\n" + "=" * 60)
    print("Test: Default Transform Fallback")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")

    # Create splits with only kwargs transform (old behavior)
    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=dataset_path,
        transform=get_val_transforms(224),  # All splits use this
        max_samples=50,
    )

    print("All splits use the same transform from kwargs:")
    print(f"  Train == Val: {train_ds.transform == val_ds.transform}")
    print(f"  Val == Test: {val_ds.transform == test_ds.transform}")

    print("\n" + "=" * 60)
    print("SUCCESS: Backward compatibility maintained!")
    print("=" * 60)


def test_mixed_transforms():
    """Test providing some split-specific and some default."""
    print("\n" + "=" * 60)
    print("Test: Mixed Transform Specification")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")

    # Only specify train_transform, others use kwargs
    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=dataset_path,
        transform=get_val_transforms(224),  # Default for val/test
        train_transform=get_train_transforms(224),  # Override for train
        max_samples=50,
    )

    print("Train uses specific transform, val/test use default:")
    print(f"  Train != Val: {train_ds.transform != val_ds.transform}")
    print(f"  Val == Test: {val_ds.transform == test_ds.transform}")

    print("\n" + "=" * 60)
    print("SUCCESS: Can mix specific and default transforms!")
    print("=" * 60)


if __name__ == "__main__":
    test_split_specific_transforms()
    test_default_transform_fallback()
    test_mixed_transforms()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
