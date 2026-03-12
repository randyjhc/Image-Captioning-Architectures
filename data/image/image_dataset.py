"""Image-only dataset for lazy loading and preprocessing."""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

from .image_utils import load_image, validate_image_path


class ImageDataset(Dataset):
    """
    Dataset for loading images with lazy loading.

    This dataset handles only image preprocessing. Images are loaded
    on-demand in __getitem__ rather than preloaded into memory.

    Args:
        image_dir: Directory containing image files
        image_filenames: List of image filenames to load
        transform: Optional transform to apply to images

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> dataset = ImageDataset(
        ...     image_dir="data/images",
        ...     image_filenames=["img1.jpg", "img2.jpg"],
        ...     transform=transform
        ... )
    """

    def __init__(
        self,
        image_dir: str | Path,
        image_filenames: list[str],
        transform: Optional[Callable] = None,
    ):
        """Initialize ImageDataset."""
        self.image_dir = Path(image_dir)
        self.image_filenames = image_filenames
        self.transform = transform

        # Validate image directory exists
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> Image.Image:
        """
        Load and return image at the given index.

        Args:
            idx: Index of the image to load

        Returns:
            Processed image (PIL Image or Tensor depending on transform)

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If image loading fails
        """
        image_filename = self.image_filenames[idx]
        image_path = validate_image_path(self.image_dir, image_filename)

        # Lazy loading: image is loaded here, not in __init__
        image = load_image(image_path, convert_mode="RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image
