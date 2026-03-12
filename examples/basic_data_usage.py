"""Example script demonstrating basic usage of Flickr8k dataset and dataloader."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import transforms

from data import create_dataloader, create_split_dataloaders

# Define image transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def example_single_dataloader():
    """Example: Create a single dataloader for the entire dataset."""
    print("=" * 60)
    print("Example 1: Single DataLoader")
    print("=" * 60)

    # Path to the dataset
    dataset_path = Path("data/datasets/flickr8k")

    # Create dataloader
    loader = create_dataloader(
        root_dir=dataset_path,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        transform=transform,
        max_samples=100,  # Limit to 100 samples for quick testing
    )

    print(f"Number of batches: {len(loader)}")
    print(f"Dataset size: {len(loader.dataset)}")

    # Iterate through first batch
    for images, captions in loader:
        print("\nFirst batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Number of captions: {len(captions)}")
        print(f"  First caption: {captions[0][:100]}...")  # Print first 100 chars
        break


def example_split_dataloaders():
    """Example: Create train/val/test dataloaders with splits."""
    print("\n" + "=" * 60)
    print("Example 2: Train/Val/Test DataLoaders")
    print("=" * 60)

    # Path to the dataset
    dataset_path = Path("data/datasets/flickr8k")

    # Create split dataloaders
    train_loader, val_loader, test_loader = create_split_dataloaders(
        root_dir=dataset_path,
        batch_size=32,
        num_workers=4,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        transform=transform,
        max_samples=100,  # Limit to 100 samples for quick testing
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print(f"\nTrain dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Iterate through first batch of each split
    for split_name, loader in [
        ("Train", train_loader),
        ("Val", val_loader),
        ("Test", test_loader),
    ]:
        for images, captions in loader:
            print(f"\n{split_name} - First batch:")
            print(f"  Images shape: {images.shape}")
            print(f"  Number of captions: {len(captions)}")
            break


def example_lazy_loading():
    """Example: Demonstrate lazy loading behavior."""
    print("\n" + "=" * 60)
    print("Example 3: Lazy Loading Demonstration")
    print("=" * 60)

    from data import FlickrDataset

    # Path to the dataset
    dataset_path = Path("data/datasets/flickr8k")

    # Create dataset (without transform for demonstration)
    dataset = FlickrDataset(
        root_dir=dataset_path,
        transform=None,  # No transform to show PIL Image
        max_samples=10,
    )

    print(f"Dataset created with {len(dataset)} samples")
    print("Images are NOT loaded into memory yet (lazy loading)")

    # Access a single sample (this triggers lazy loading for that sample only)
    print("\nAccessing sample 0...")
    image, caption = dataset[0]

    print(f"Image type: {type(image)}")
    print(f"Image size: {image.size}")
    print(f"Caption: {caption[:100]}...")

    # Get image filename without loading
    print(f"\nImage filename: {dataset.get_image_filename(0)}")
    print("(Filename retrieved without loading the image)")


if __name__ == "__main__":
    # Run all examples
    example_single_dataloader()
    example_split_dataloaders()
    example_lazy_loading()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
