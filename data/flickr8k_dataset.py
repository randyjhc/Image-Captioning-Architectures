import csv
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from Image_Captioning_Architectures.data.vocab import Vocabulary


class Flickr8kDataset(Dataset):
    """
    Each dataset item returns one (image, caption) pair.
    Since Flickr8k has multiple captions per image, the same image filename
    will appear multiple times with different captions.
    """

    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        vocab: Vocabulary,
        transform=None,
    ):
        self.image_dir = image_dir
        self.captions_file = captions_file
        self.vocab = vocab
        self.transform = transform

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        with open(self.captions_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Expected header: image,caption
            if len(header) < 2 or header[0].strip().lower() != "image":
                raise ValueError(
                    f"Unexpected captions header: {header}. Expected ['image', 'caption']"
                )

            for row in reader:
                if len(row) < 2:
                    continue

                image_name = row[0].strip()
                caption = row[1].strip()

                image_path = os.path.join(self.image_dir, image_name)
                if not os.path.exists(image_path):
                    continue

                samples.append((image_path, caption))

        if len(samples) == 0:
            raise ValueError("No valid image-caption samples found.")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        token_ids = [self.vocab.sos_id]
        token_ids += self.vocab.numericalize(caption)
        token_ids += [self.vocab.eos_id]

        caption_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "image": image,
            "caption_ids": caption_tensor,
            "raw_caption": caption,
            "image_path": image_path,
        }


def build_vocab_from_captions(
    captions_file: str,
    min_freq: int = 1,
) -> Vocabulary:
    captions = []

    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:
                continue
            captions.append(row[1].strip())

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(captions)
    return vocab


def make_flickr8k_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def flickr8k_collate_fn(batch, pad_id: int):
    images = [item["image"] for item in batch]
    caption_ids = [item["caption_ids"] for item in batch]
    raw_captions = [item["raw_caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    images = torch.stack(images, dim=0)
    captions = pad_sequence(caption_ids, batch_first=True, padding_value=pad_id)

    return {
        "images": images,
        "captions": captions,
        "raw_captions": raw_captions,
        "image_paths": image_paths,
    }


def split_samples_by_image(
    captions_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split by image name so all captions of the same image stay in the same split.
    Returns sets of image filenames.
    """
    image_to_captions = defaultdict(list)

    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            if len(row) < 2:
                continue
            image_name = row[0].strip()
            caption = row[1].strip()
            image_to_captions[image_name].append(caption)

    image_names = list(image_to_captions.keys())
    rng = random.Random(seed)
    rng.shuffle(image_names)

    n = len(image_names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_images = set(image_names[:n_train])
    val_images = set(image_names[n_train : n_train + n_val])
    test_images = set(image_names[n_train + n_val :])

    return train_images, val_images, test_images


class Flickr8kSubsetDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        vocab: Vocabulary,
        allowed_images: set[str],
        transform=None,
    ):
        self.image_dir = image_dir
        self.captions_file = captions_file
        self.vocab = vocab
        self.transform = transform
        self.allowed_images = allowed_images

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        with open(self.captions_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if len(row) < 2:
                    continue

                image_name = row[0].strip()
                caption = row[1].strip()

                if image_name not in self.allowed_images:
                    continue

                image_path = os.path.join(self.image_dir, image_name)
                if not os.path.exists(image_path):
                    continue

                samples.append((image_path, caption))

        if len(samples) == 0:
            raise ValueError("No samples found for this split.")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        token_ids = [self.vocab.sos_id]
        token_ids += self.vocab.numericalize(caption)
        token_ids += [self.vocab.eos_id]

        return {
            "image": image,
            "caption_ids": torch.tensor(token_ids, dtype=torch.long),
            "raw_caption": caption,
            "image_path": image_path,
        }


def build_flickr8k_dataloaders(
    data_root: str,
    batch_size: int = 16,
    image_size: int = 224,
    min_freq: int = 1,
    num_workers: int = 0,
):
    image_dir = os.path.join(data_root, "Images")
    captions_file = os.path.join(data_root, "captions.txt")

    vocab = build_vocab_from_captions(captions_file, min_freq=min_freq)
    transform = make_flickr8k_transforms(image_size=image_size)

    train_images, val_images, test_images = split_samples_by_image(captions_file)

    train_dataset = Flickr8kSubsetDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        vocab=vocab,
        allowed_images=train_images,
        transform=transform,
    )
    val_dataset = Flickr8kSubsetDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        vocab=vocab,
        allowed_images=val_images,
        transform=transform,
    )
    test_dataset = Flickr8kSubsetDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        vocab=vocab,
        allowed_images=test_images,
        transform=transform,
    )

    def collate(batch):
        return flickr8k_collate_fn(batch, pad_id=vocab.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader, vocab
