"""Utility functions for image loading and preprocessing."""

from pathlib import Path

from PIL import Image


def load_image(image_path: str | Path, convert_mode: str = "RGB") -> Image.Image:
    """
    Lazily load an image from disk.

    Args:
        image_path: Path to the image file
        convert_mode: PIL image mode to convert to (default: "RGB")

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If the image file doesn't exist
        Exception: If the image is corrupted or cannot be opened
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        if convert_mode:
            image = image.convert(convert_mode)
        return image
    except Exception as e:
        raise Exception(f"Failed to load image {image_path}: {str(e)}")


def validate_image_path(image_dir: str | Path, image_filename: str) -> Path:
    """
    Validate and construct full image path.

    Args:
        image_dir: Directory containing images
        image_filename: Name of the image file

    Returns:
        Full path to the image as Path object

    Raises:
        ValueError: If image_dir doesn't exist
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    return image_dir / image_filename
