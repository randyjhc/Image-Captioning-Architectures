"""Advanced usage examples for Flickr8k dataset and dataloader."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import transforms

from data import FlickrDataset, create_split_dataloaders
from data.image.transforms import (
    MINIMAL_TRANSFORMS,
    TRAIN_TRANSFORMS_224,
    VAL_TRANSFORMS_224,
    get_custom_transforms,
    get_train_transforms,
    get_val_transforms,
)


def example_using_presets():
    """Example: Using predefined transform presets."""
    print("=" * 60)
    print("Example 1: Using Predefined Transform Presets")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")

    # Use ready-to-use presets
    train_dataset = FlickrDataset(
        root_dir=dataset_path, transform=TRAIN_TRANSFORMS_224, max_samples=10
    )

    val_dataset = FlickrDataset(
        root_dir=dataset_path, transform=VAL_TRANSFORMS_224, max_samples=10
    )

    print(f"Train dataset with augmentation: {len(train_dataset)} samples")
    print(f"Val dataset without augmentation: {len(val_dataset)} samples")

    # Get a sample
    train_img, caption = train_dataset[0]
    val_img, _ = val_dataset[0]

    print(f"\nTrain image shape: {train_img.shape}")
    print(f"Val image shape: {val_img.shape}")
    print(f"Caption: {caption[:80]}...")

    print("\nTrain transforms:")
    print(TRAIN_TRANSFORMS_224)
    print("\nVal transforms:")
    print(VAL_TRANSFORMS_224)


def example_custom_factory():
    """Example: Using transform factory functions with custom parameters."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Transform Parameters")
    print("=" * 60)

    # Create transforms with custom parameters
    train_transform = get_train_transforms(
        image_size=224,
        resize_size=256,
        horizontal_flip_prob=0.7,  # More aggressive flipping
        color_jitter_brightness=0.3,  # Stronger color jitter
        color_jitter_contrast=0.3,
    )

    val_transform = get_val_transforms(image_size=224)

    print("Custom train transform with aggressive augmentation:")
    print(train_transform)
    print("\nStandard val transform:")
    print(val_transform)


def example_with_dataloaders():
    """Example: Using transforms with dataloaders."""
    print("\n" + "=" * 60)
    print("Example 3: Transforms with DataLoaders")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")

    # Create splits with different transforms
    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=dataset_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        max_samples=100,
    )

    # Apply appropriate transforms to each split
    train_ds.transform = get_train_transforms(image_size=224)  # With augmentation
    val_ds.transform = get_val_transforms(image_size=224)  # Without augmentation
    test_ds.transform = get_val_transforms(image_size=224)  # Without augmentation

    print(f"Train dataset: {len(train_ds)} samples with augmentation")
    print(f"Val dataset: {len(val_ds)} samples without augmentation")
    print(f"Test dataset: {len(test_ds)} samples without augmentation")

    # Get samples from each split
    train_img, _ = train_ds[0]
    val_img, _ = val_ds[0]
    test_img, _ = test_ds[0]

    print("\nAll images are normalized to 224x224:")
    print(f"  Train: {train_img.shape}")
    print(f"  Val: {val_img.shape}")
    print(f"  Test: {test_img.shape}")


def example_custom_normalization():
    """Example: Custom normalization for different pretrained models."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Normalization")
    print("=" * 60)

    # ImageNet normalization (default)
    _ = get_train_transforms(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # Custom normalization (e.g., for models trained on different data)
    _ = get_train_transforms(
        image_size=224,
        mean=(0.5, 0.5, 0.5),  # Custom mean
        std=(0.5, 0.5, 0.5),  # Custom std
    )

    # No normalization (just scale to [0, 1])
    _ = get_custom_transforms(image_size=224, augment=True, normalize=False)

    print("ImageNet normalization (default):")
    print("  Mean: (0.485, 0.456, 0.406)")
    print("  Std: (0.229, 0.224, 0.225)")

    print("\nCustom normalization:")
    print("  Mean: (0.5, 0.5, 0.5)")
    print("  Std: (0.5, 0.5, 0.5)")

    print("\nNo normalization:")
    print("  Images scaled to [0, 1] only")


def example_minimal_transforms():
    """Example: Using minimal transforms for visualization."""
    print("\n" + "=" * 60)
    print("Example 5: Minimal Transforms for Visualization")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")

    # Use minimal transforms (no normalization, just resize + to_tensor)
    dataset = FlickrDataset(
        root_dir=dataset_path, transform=MINIMAL_TRANSFORMS, max_samples=5
    )

    print(f"Dataset with minimal transforms: {len(dataset)} samples")
    print("\nMinimal transform (good for visualization):")
    print(MINIMAL_TRANSFORMS)

    # Get a sample
    image, caption = dataset[0]
    print(f"\nImage shape: {image.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Caption: {caption[:80]}...")
    print("\nNote: Image values are in [0, 1] without normalization")
    print("This is ideal for visualization or models that expect unnormalized inputs")


def example_dataset_statistics():
    """Example: Analyze dataset statistics."""
    print("\n" + "=" * 60)
    print("Example 6: Dataset Statistics")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")
    dataset = FlickrDataset(root_dir=dataset_path)

    print(f"Total samples: {len(dataset)}")

    # Count unique images (each image has 5 captions)
    unique_images = set(dataset.get_image_filename(i) for i in range(len(dataset)))
    print(f"Unique images: {len(unique_images)}")
    print(f"Captions per image: {len(dataset) / len(unique_images):.1f}")

    # Analyze caption lengths
    caption_lengths = [len(dataset.get_caption(i).split()) for i in range(100)]
    print("\nCaption length statistics (first 100):")
    print(f"  Min: {min(caption_lengths)} words")
    print(f"  Max: {max(caption_lengths)} words")
    print(f"  Mean: {sum(caption_lengths) / len(caption_lengths):.1f} words")


def example_custom_tokenizer():
    """Example: Using a custom tokenizer interface."""
    print("\n" + "=" * 60)
    print("Example 7: Custom Tokenizer Interface")
    print("=" * 60)

    # Mock tokenizer (replace with real tokenizer from text modules)
    class SimpleTokenizer:
        def __init__(self):
            self.word_to_idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
            self.idx = 4

        def encode(self, text):
            """Simple word-level tokenization."""
            words = text.lower().split()
            encoded = [1]  # <START>
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.idx
                    self.idx += 1
                encoded.append(self.word_to_idx[word])
            encoded.append(2)  # <END>
            return encoded

    tokenizer = SimpleTokenizer()

    dataset_path = Path("data/datasets/flickr8k")
    dataset = FlickrDataset(root_dir=dataset_path, tokenizer=tokenizer, max_samples=10)

    # Get a sample with tokenized caption
    image, caption_tokens = dataset[0]

    print(f"Original caption: {dataset.get_caption(0)}")
    print(f"Tokenized caption: {caption_tokens}")
    print(f"Number of tokens: {len(caption_tokens)}")
    print(f"Vocabulary size: {len(tokenizer.word_to_idx)}")


def example_custom_collate():
    """Example: Using custom collate function with padding."""
    print("\n" + "=" * 60)
    print("Example 8: Custom Collate Function with Padding")
    print("=" * 60)

    # Mock tokenizer
    class SimpleTokenizer:
        def encode(self, text):
            return [1] + [hash(w) % 1000 for w in text.split()[:10]] + [2]

    tokenizer = SimpleTokenizer()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset_path = Path("data/datasets/flickr8k")

    # Create dataloader with padding collate function
    train_loader, _, _ = create_split_dataloaders(
        root_dir=dataset_path,
        batch_size=8,
        num_workers=2,
        transform=transform,
        tokenizer=tokenizer,
        collate_fn_type="padding",  # Enable padding
        pad_token_id=0,
        max_samples=50,
    )

    # Check batch
    for images, captions in train_loader:
        print(f"Images shape: {images.shape}")
        print(f"Captions shape: {captions.shape}")
        print(f"Captions (padded tensor):\n{captions}")
        print(
            f"\nAll captions have same length: {all(len(c) == len(captions[0]) for c in captions)}"
        )
        break


if __name__ == "__main__":
    # Run all examples
    example_using_presets()
    example_custom_factory()
    example_with_dataloaders()
    example_custom_normalization()
    example_minimal_transforms()
    example_dataset_statistics()
    example_custom_tokenizer()
    example_custom_collate()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
