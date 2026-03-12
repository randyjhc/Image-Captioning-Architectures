# Examples

This directory contains example scripts demonstrating how to use the Flickr8k dataset and dataloader.

## Available Examples

### 1. [basic_data_usage.py](basic_data_usage.py)
Basic examples showing fundamental usage patterns:
- Creating a single dataloader
- Creating train/val/test dataloaders with splits
- Demonstrating lazy loading behavior

**Run it:**
```bash
python examples/basic_data_usage.py
```

### 2. [advanced_data_usage.py](advanced_data_usage.py)
Comprehensive examples covering both transforms and advanced features:

**Transform Examples:**
- Using predefined transform presets (`TRAIN_TRANSFORMS_224`, `VAL_TRANSFORMS_224`)
- Customizing transform parameters with factory functions
- Applying transforms to train/val/test splits
- Custom normalization for different pretrained models
- Minimal transforms for visualization

**Advanced Features:**
- Dataset statistics analysis
- Custom tokenizer interface
- Custom collate functions with padding

**Run it:**
```bash
python examples/advanced_data_usage.py
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

# Create splits with different transforms
train_ds, val_ds, test_ds = FlickrDataset.create_splits(
    root_dir="data/datasets/flickr8k",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    train_transform=get_train_transforms(image_size=224),  # With augmentation
    val_transform=get_val_transforms(image_size=224),      # Without augmentation
    test_transform=get_val_transforms(image_size=224),     # Without augmentation
)
```

### Using Predefined Transform Presets

```python
from data import FlickrDataset
from data.image.transforms import TRAIN_TRANSFORMS_224, VAL_TRANSFORMS_224

# Use ready-to-use presets
train_dataset = FlickrDataset(
    root_dir="data/datasets/flickr8k",
    transform=TRAIN_TRANSFORMS_224
)

val_dataset = FlickrDataset(
    root_dir="data/datasets/flickr8k",
    transform=VAL_TRANSFORMS_224
)
```

## Running All Examples

To run all examples at once:

```bash
python examples/basic_data_usage.py
python examples/advanced_data_usage.py
```

## Example Overview

| Example | Focus | Difficulty |
|---------|-------|-----------|
| `basic_data_usage.py` | DataLoader basics, lazy loading | Beginner |
| `advanced_data_usage.py` | Transforms + advanced features | Intermediate |

## Notes

- All examples use `max_samples` parameter to limit dataset size for quick testing
- Remove `max_samples` parameter to use the full dataset (40,455 samples)
- The dataset path is assumed to be `data/datasets/flickr8k/`
- Ensure you have downloaded the dataset first (see main README.md)

## Transform System

The examples demonstrate the comprehensive transform system available:

**Predefined Presets:**
- `TRAIN_TRANSFORMS_224` - Training with augmentation (224x224)
- `VAL_TRANSFORMS_224` - Validation without augmentation (224x224)
- `MINIMAL_TRANSFORMS` - Minimal processing for visualization

**Factory Functions:**
- `get_train_transforms(image_size, ...)` - Customizable training transforms
- `get_val_transforms(image_size, ...)` - Customizable validation transforms
- `get_custom_transforms(...)` - Full control over all parameters

See [advanced_data_usage.py](advanced_data_usage.py) for detailed examples of each approach.
