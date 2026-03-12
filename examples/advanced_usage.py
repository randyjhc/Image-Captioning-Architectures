"""Advanced usage examples for Flickr8k dataset and dataloader."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import transforms

from data import FlickrDataset, create_split_dataloaders


def example_custom_transforms():
    """Example: Using different transforms for train and test."""
    print("=" * 60)
    print("Example 1: Custom Transforms for Train/Test")
    print("=" * 60)

    # Training transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Test transforms without augmentation
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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

    # Apply different transforms
    train_ds.transform = train_transform
    val_ds.transform = test_transform
    test_ds.transform = test_transform

    print(f"Train dataset: {len(train_ds)} samples with augmentation")
    print(f"Val dataset: {len(val_ds)} samples without augmentation")
    print(f"Test dataset: {len(test_ds)} samples without augmentation")

    # Test transforms
    train_img, _ = train_ds[0]
    val_img, _ = val_ds[0]

    print(f"\nTrain image shape: {train_img.shape}")
    print(f"Val image shape: {val_img.shape}")


def example_dataset_statistics():
    """Example: Analyze dataset statistics."""
    print("\n" + "=" * 60)
    print("Example 2: Dataset Statistics")
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
    print("Example 3: Custom Tokenizer Interface")
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
    print("Example 4: Custom Collate Function with Padding")
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


def example_memory_efficient_iteration():
    """Example: Memory-efficient iteration with lazy loading."""
    print("\n" + "=" * 60)
    print("Example 5: Memory-Efficient Iteration")
    print("=" * 60)

    dataset_path = Path("data/datasets/flickr8k")

    # Create dataset without transform (minimal memory)
    dataset = FlickrDataset(root_dir=dataset_path, max_samples=1000)

    print(f"Dataset created with {len(dataset)} samples")
    print("Memory usage is minimal - images not loaded yet")

    # Iterate through dataset - images loaded one at a time
    import time

    start = time.time()
    for i in range(10):
        image, caption = dataset[i]
        # Process image here...
        pass
    elapsed = time.time() - start

    print(f"\nProcessed 10 images in {elapsed:.2f}s")
    print("Each image was loaded, processed, and freed from memory")


if __name__ == "__main__":
    # Run all examples
    example_custom_transforms()
    example_dataset_statistics()
    example_custom_tokenizer()
    example_custom_collate()
    example_memory_efficient_iteration()

    print("\n" + "=" * 60)
    print("All advanced examples completed!")
    print("=" * 60)
