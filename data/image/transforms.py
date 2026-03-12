"""Predefined image transformations for training and inference."""

from typing import Callable

from torchvision import transforms


def get_train_transforms(
    image_size: int = 224,
    resize_size: int = 256,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    horizontal_flip_prob: float = 0.5,
    color_jitter_brightness: float = 0.2,
    color_jitter_contrast: float = 0.2,
    color_jitter_saturation: float = 0.2,
    color_jitter_hue: float = 0.1,
) -> Callable:
    """
    Get training transforms with data augmentation.

    Applies random augmentations to increase model robustness:
    - Random resized crop
    - Random horizontal flip
    - Color jitter (brightness, contrast, saturation, hue)
    - Normalization using ImageNet statistics by default

    Args:
        image_size: Target image size (square)
        resize_size: Initial resize before random crop
        mean: Normalization mean for each channel (RGB)
        std: Normalization std for each channel (RGB)
        horizontal_flip_prob: Probability of horizontal flip
        color_jitter_brightness: Brightness jitter factor
        color_jitter_contrast: Contrast jitter factor
        color_jitter_saturation: Saturation jitter factor
        color_jitter_hue: Hue jitter factor

    Returns:
        Composed transform callable

    Example:
        >>> transform = get_train_transforms(image_size=224)
        >>> dataset = FlickrDataset(root_dir="data/flickr8k", transform=transform)
    """
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.ColorJitter(
                brightness=color_jitter_brightness,
                contrast=color_jitter_contrast,
                saturation=color_jitter_saturation,
                hue=color_jitter_hue,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_val_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable:
    """
    Get validation/test transforms without augmentation.

    Applies deterministic preprocessing:
    - Resize to target size
    - Convert to tensor
    - Normalization using ImageNet statistics by default

    Args:
        image_size: Target image size (square)
        mean: Normalization mean for each channel (RGB)
        std: Normalization std for each channel (RGB)

    Returns:
        Composed transform callable

    Example:
        >>> transform = get_val_transforms(image_size=224)
        >>> dataset = FlickrDataset(root_dir="data/flickr8k", transform=transform)
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# Alias for get_val_transforms for clarity in inference code
get_inference_transforms = get_val_transforms


def get_custom_transforms(
    image_size: int = 224,
    augment: bool = True,
    normalize: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    **augmentation_params,
) -> Callable:
    """
    Get custom transforms with flexible configuration.

    Args:
        image_size: Target image size (square)
        augment: Whether to apply data augmentation
        normalize: Whether to apply normalization
        mean: Normalization mean for each channel (RGB)
        std: Normalization std for each channel (RGB)
        **augmentation_params: Additional augmentation parameters
            (horizontal_flip_prob, color_jitter_brightness, etc.)

    Returns:
        Composed transform callable

    Example:
        >>> # Training with augmentation
        >>> transform = get_custom_transforms(
        ...     image_size=224,
        ...     augment=True,
        ...     horizontal_flip_prob=0.7
        ... )
        >>> # Validation without normalization
        >>> transform = get_custom_transforms(
        ...     image_size=224,
        ...     augment=False,
        ...     normalize=False
        ... )
    """
    transform_list = []

    if augment:
        resize_size = augmentation_params.get("resize_size", 256)
        transform_list.extend(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(
                    p=augmentation_params.get("horizontal_flip_prob", 0.5)
                ),
                transforms.ColorJitter(
                    brightness=augmentation_params.get("color_jitter_brightness", 0.2),
                    contrast=augmentation_params.get("color_jitter_contrast", 0.2),
                    saturation=augmentation_params.get("color_jitter_saturation", 0.2),
                    hue=augmentation_params.get("color_jitter_hue", 0.1),
                ),
            ]
        )
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


# Predefined transform presets
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Standard 224x224 transforms
TRAIN_TRANSFORMS_224 = get_train_transforms(image_size=224)
VAL_TRANSFORMS_224 = get_val_transforms(image_size=224)
INFERENCE_TRANSFORMS_224 = get_inference_transforms(image_size=224)

# Standard 299x299 transforms (for Inception-based models)
TRAIN_TRANSFORMS_299 = get_train_transforms(image_size=299, resize_size=320)
VAL_TRANSFORMS_299 = get_val_transforms(image_size=299)
INFERENCE_TRANSFORMS_299 = get_inference_transforms(image_size=299)

# Minimal transforms (no normalization)
MINIMAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
