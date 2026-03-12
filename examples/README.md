# Examples

This directory contains example scripts demonstrating how to use the Flickr8k dataset and dataloader.

## Available Examples

### 1. [basic_usage.py](basic_usage.py)
Basic examples showing fundamental usage patterns:
- Creating a single dataloader
- Creating train/val/test dataloaders with splits
- Demonstrating lazy loading behavior

**Run it:**
```bash
python examples/basic_usage.py
```

### 2. [advanced_usage.py](advanced_usage.py)
Advanced usage patterns including:
- Custom transforms for train/test
- Dataset statistics analysis
- Custom tokenizer interface
- Custom collate functions with padding
- Memory-efficient iteration

**Run it:**
```bash
python examples/advanced_usage.py
```

### 3. [using_transforms.py](using_transforms.py)
Comprehensive examples of the transform system:
- Using predefined transform presets
- Customizing transform parameters
- Different image sizes for different models
- Transforms with dataloaders
- Custom normalization
- Minimal transforms for visualization

**Run it:**
```bash
python examples/using_transforms.py
```

## Quick Start

### Minimal Example

```python
from data import create_dataloader
from data.image.transforms import TRAIN_TRANSFORMS_224

# Create a dataloader with predefined transforms
loader = create_dataloader(
    root_dir="data/datasets/flickr8k",
    batch_size=32,
    transform=TRAIN_TRANSFORMS_224
)

# Iterate through batches
for images, captions in loader:
    print(f"Batch: {images.shape}, {len(captions)} captions")
    break
```

### Train/Val/Test Splits

```python
from data import FlickrDataset
from data.image.transforms import get_train_transforms, get_val_transforms

# Create splits
train_ds, val_ds, test_ds = FlickrDataset.create_splits(
    root_dir="data/datasets/flickr8k",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)

# Apply different transforms
train_ds.transform = get_train_transforms(image_size=224)  # With augmentation
val_ds.transform = get_val_transforms(image_size=224)      # Without augmentation
test_ds.transform = get_val_transforms(image_size=224)     # Without augmentation
```

## Running All Examples

To run all examples at once:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
python examples/using_transforms.py
```

## Notes

- All examples use `max_samples` parameter to limit dataset size for quick testing
- Remove `max_samples` parameter to use the full dataset (40,455 samples)
- The dataset path is assumed to be `data/datasets/flickr8k/`
- Ensure you have downloaded the dataset first (see main README.md)
