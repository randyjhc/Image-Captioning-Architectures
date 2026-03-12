# Data Module

This module provides dataset and dataloader implementations for the Flickr8k image captioning dataset with lazy loading and modular architecture.

## Architecture

```
data/
├── image/
│   ├── __init__.py
│   ├── image_dataset.py      # Image-only dataset with lazy loading
│   ├── image_utils.py         # Utility functions for image loading
│   └── transforms.py          # Predefined image transformation presets
├── text/
│   ├── clean_caption.py       # (Future) Caption preprocessing
│   ├── vocab.py               # (Future) Vocabulary building
│   └── tokenizer.py           # (Future) Text tokenization
├── __init__.py
├── flickr_dataset.py          # Main Flickr8k dataset class
└── dataloader.py              # DataLoader factory functions
```

## Features

### 1. Lazy Loading
Images are loaded on-demand in `__getitem__` rather than preloaded into memory:

```python
from data import FlickrDataset

# Dataset is created instantly without loading any images
dataset = FlickrDataset(root_dir="data/datasets/flickr8k")

# Image is loaded only when accessed
image, caption = dataset[0]  # Loads image from disk here
```

### 2. Modular Image Processing
Image preprocessing is separated into dedicated modules:

```python
from data.image import ImageDataset, load_image
from data.image.transforms import get_train_transforms, get_val_transforms

# Use predefined transforms
train_transform = get_train_transforms(image_size=224)
val_transform = get_val_transforms(image_size=224)

# Use standalone image utilities
image = load_image("path/to/image.jpg")

# Use image-only dataset
image_dataset = ImageDataset(
    image_dir="data/images",
    image_filenames=["img1.jpg", "img2.jpg"],
    transform=train_transform
)
```

### 3. Text Processing Interface
Text processing interfaces are preserved but not implemented:

```python
# Text interface (implementation in text modules)
dataset = FlickrDataset(
    root_dir="data/datasets/flickr8k",
    tokenizer=tokenizer,  # Will be used when text modules are ready
)
```

### 4. Train/Val/Test Splits
Easy dataset splitting based on unique images with support for different transforms per split:

```python
from data import FlickrDataset
from data.image.transforms import get_train_transforms, get_val_transforms

# Method 1: Different transforms per split (recommended)
train_ds, val_ds, test_ds = FlickrDataset.create_splits(
    root_dir="data/datasets/flickr8k",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    train_transform=get_train_transforms(224),  # With augmentation
    val_transform=get_val_transforms(224),      # Without augmentation
    test_transform=get_val_transforms(224),     # Without augmentation
)

# Method 2: Same transform for all splits
train_ds, val_ds, test_ds = FlickrDataset.create_splits(
    root_dir="data/datasets/flickr8k",
    transform=get_val_transforms(224),  # All splits use this
)
```

### 5. Predefined Image Transforms
Standard transformations for training and inference:

```python
from data.image.transforms import (
    get_train_transforms,
    get_val_transforms,
    TRAIN_TRANSFORMS_224,
    VAL_TRANSFORMS_224,
)

# Get customizable transforms
train_transform = get_train_transforms(image_size=224, horizontal_flip_prob=0.7)
val_transform = get_val_transforms(image_size=224)

# Or use predefined presets
dataset = FlickrDataset(
    root_dir="data/datasets/flickr8k",
    transform=TRAIN_TRANSFORMS_224  # Ready-to-use preset
)
```

### 6. DataLoader Factory Functions
Convenient dataloader creation:

```python
from data import create_dataloader, create_split_dataloaders
from data.image.transforms import get_train_transforms, get_val_transforms

# Get transforms
train_transform = get_train_transforms(image_size=224)
val_transform = get_val_transforms(image_size=224)

# Single dataloader
loader = create_dataloader(
    root_dir="data/datasets/flickr8k",
    batch_size=32,
    transform=train_transform
)

# Split dataloaders with different transforms
train_loader, val_loader, test_loader = create_split_dataloaders(
    root_dir="data/datasets/flickr8k",
    batch_size=32,
    transform=train_transform  # Will be overridden per split if needed
)
```

## Usage Examples

### Basic Usage

```python
from data import create_dataloader
from data.image.transforms import get_train_transforms

# Get predefined transform
transform = get_train_transforms(image_size=224)

# Create dataloader
loader = create_dataloader(
    root_dir="data/datasets/flickr8k",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    transform=transform
)

# Iterate through batches
for images, captions in loader:
    # images: torch.Tensor [batch_size, 3, 224, 224]
    # captions: list[str] of length batch_size
    print(f"Batch: {images.shape}, {len(captions)} captions")
```

### With Train/Val/Test Splits

```python
from data import create_split_dataloaders, FlickrDataset
from data.image.transforms import get_train_transforms, get_val_transforms

# Method 1: Using factory function (same transform for all splits)
train_loader, val_loader, test_loader = create_split_dataloaders(
    root_dir="data/datasets/flickr8k",
    batch_size=32,
    num_workers=4,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    transform=get_train_transforms(image_size=224)
)

# Method 2: Different transforms for train/val/test
train_ds, val_ds, test_ds = FlickrDataset.create_splits(
    root_dir="data/datasets/flickr8k",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)

# Apply different transforms
train_ds.transform = get_train_transforms(image_size=224)  # With augmentation
val_ds.transform = get_val_transforms(image_size=224)      # Without augmentation
test_ds.transform = get_val_transforms(image_size=224)     # Without augmentation

# Training loop
for epoch in range(num_epochs):
    for images, captions in train_loader:
        # Training code here
        pass
```

### Custom Collate Function (for Tokenized Captions)

```python
from data import create_dataloader

# When using tokenized captions with padding
loader = create_dataloader(
    root_dir="data/datasets/flickr8k",
    batch_size=32,
    transform=transform,
    tokenizer=my_tokenizer,  # Custom tokenizer
    collate_fn_type="padding",  # Pad captions to same length
    pad_token_id=0  # Padding token ID
)

# Now captions are padded tensors
for images, captions in loader:
    # images: torch.Tensor [batch_size, 3, 224, 224]
    # captions: torch.Tensor [batch_size, max_seq_len]
    pass
```

### Direct Dataset Access

```python
from data import FlickrDataset

dataset = FlickrDataset(
    root_dir="data/datasets/flickr8k",
    transform=transform
)

# Access individual samples
image, caption = dataset[0]

# Get metadata without loading image
filename = dataset.get_image_filename(0)
caption_text = dataset.get_caption(0)

# Dataset statistics
print(f"Total samples: {len(dataset)}")
```

## API Reference

### FlickrDataset

Main dataset class for Flickr8k.

**Constructor:**
```python
FlickrDataset(
    root_dir: str | Path,
    captions_file: str = "captions.txt",
    transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
    max_samples: Optional[int] = None
)
```

**Methods:**
- `__len__()`: Return number of samples
- `__getitem__(idx)`: Get image and caption at index
- `get_image_filename(idx)`: Get image filename without loading
- `get_caption(idx)`: Get raw caption text
- `create_splits(...)`: Static method to create train/val/test splits

### ImageDataset

Image-only dataset with lazy loading.

**Constructor:**
```python
ImageDataset(
    image_dir: str | Path,
    image_filenames: list[str],
    transform: Optional[Callable] = None
)
```

### DataLoader Functions

**create_dataloader:**
```python
create_dataloader(
    root_dir: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Any] = None,
    collate_fn_type: str = "default",
    pad_token_id: int = 0,
    max_samples: Optional[int] = None,
    **dataset_kwargs
) -> DataLoader
```

**create_split_dataloaders:**
```python
create_split_dataloaders(
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
    **dataset_kwargs
) -> tuple[DataLoader, DataLoader, DataLoader]
```

### Transform Functions

**transforms.py:**
```python
# Transform factory functions
get_train_transforms(
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    horizontal_flip_prob: float = 0.5,
    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    color_jitter_saturation: float = 0.2,
    color_jitter_hue: float = 0.1,
) -> Callable

get_val_transforms(
    image_size: int = 224,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> Callable

get_inference_transforms(
    image_size: int = 224,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> Callable

get_custom_transforms(
    image_size: int = 224,
    augment: bool = True,
    normalize: bool = True,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    **augmentation_params
) -> Callable

# Predefined presets
TRAIN_TRANSFORMS_224    # 224x224 training transforms
VAL_TRANSFORMS_224      # 224x224 validation transforms
INFERENCE_TRANSFORMS_224  # 224x224 inference transforms
TRAIN_TRANSFORMS_299    # 299x299 training (for Inception)
VAL_TRANSFORMS_299      # 299x299 validation
INFERENCE_TRANSFORMS_299  # 299x299 inference
MINIMAL_TRANSFORMS      # Basic resize + to_tensor (no normalization)
```

**image_utils.py:**
```python
load_image(image_path: str | Path, convert_mode: str = "RGB") -> Image.Image
validate_image_path(image_dir: str | Path, image_filename: str) -> Path
```

## Design Decisions

1. **Lazy Loading**: Images are loaded in `__getitem__` to minimize memory usage
2. **Modular Architecture**: Separate modules for image and text processing
3. **Predefined Transforms**: Ready-to-use transformation presets for common use cases
4. **Interface Preservation**: Text processing interfaces defined but not implemented
5. **Type Hints**: Full type annotations for better IDE support
6. **Error Handling**: Graceful handling of missing/corrupted images
7. **Flexibility**: Support for custom transforms, tokenizers, and collate functions

## Future Enhancements

The following modules are planned but not yet implemented:

- `text/clean_caption.py`: Caption preprocessing and cleaning
- `text/vocab.py`: Vocabulary building from captions
- `text/tokenizer.py`: Text tokenization and encoding

These interfaces are already preserved in the current implementation and can be added without breaking changes.
