"""Entry point for image captioning training and generation.

Usage:
    python image_caption.py [cnn|vit] [train|gen] [--config CONFIG]

Examples:
    python image_caption.py vit train --config configs/vit_train.json
    python image_caption.py vit gen   --config configs/vit_gen.json
    python image_caption.py vit train  # uses ConfigViT defaults
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import logger_setup

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_json_config(path: str | None) -> dict[str, Any]:
    """Load a JSON config file and return the raw dict. Returns {} if path is None."""
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a JSON object, got {type(cfg).__name__}")
    return cfg


# ---------------------------------------------------------------------------
# ViT handlers
# ---------------------------------------------------------------------------


def run_train_vit(cfg_dict: dict[str, Any]) -> None:
    """Train the ViT image captioning model."""
    from train_vit.config import ConfigViT
    from train_vit.trainer import TrainerViT

    config = ConfigViT.from_dict(cfg_dict)
    logger_setup(log_file=config.log_file)
    trainer = TrainerViT(config, config.data_root, checkpoint_dir=config.checkpoint_dir)
    trainer.fit()


def run_gen_vit(cfg_dict: dict[str, Any]) -> None:
    """Generate captions for images using a trained ViT model."""
    import torch
    from PIL import Image

    from data.image.transforms import get_val_transforms
    from model_vit.generator import GeneratorViT
    from train_vit.config import ConfigViT

    config = ConfigViT.from_dict(cfg_dict)
    checkpoint_path = cfg_dict.get(
        "checkpoint_path", Path(config.checkpoint_dir) / "best.pt"
    )
    image_paths = cfg_dict.get("image_paths", config.image_paths)
    max_len: int = cfg_dict.get("max_len", 30)

    if not image_paths:
        raise ValueError("'image_paths' must be a non-empty list.")

    generator = GeneratorViT.from_checkpoint(checkpoint_path, config.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.model.to(device)

    assert generator.config is not None
    transform = get_val_transforms(generator.config.image_size)
    images = torch.stack(
        [transform(Image.open(p).convert("RGB")) for p in image_paths]
    ).to(device)

    captions = generator.generate_caption(images, max_len=max_len)
    for path, caption in zip(image_paths, captions):
        print(f"{Path(path).name}: {caption}")


# ---------------------------------------------------------------------------
# CNN handlers (not yet implemented)
# ---------------------------------------------------------------------------


def run_train_cnn(cfg_dict: dict[str, Any]) -> None:
    from train_cnn.config import ConfigCNN
    from train_cnn.trainer import TrainerCNN

    config = ConfigCNN.from_dict(cfg_dict)
    logger_setup(log_file=config.log_file)
    trainer = TrainerCNN(config, config.data_root, checkpoint_dir=config.checkpoint_dir)
    trainer.fit()


def run_gen_cnn(cfg_dict: dict[str, Any]) -> None:
    import torch
    from PIL import Image

    from data.image.transforms import get_val_transforms
    from train_cnn.config import ConfigCNN
    from train_cnn.trainer import TrainerCNN

    config = ConfigCNN.from_dict(cfg_dict)
    checkpoint_path = cfg_dict.get(
        "checkpoint_path", Path(config.checkpoint_dir) / "best.pt"
    )
    image_paths = cfg_dict.get("image_paths", config.image_paths)
    max_len: int = cfg_dict.get("max_len", 30)

    if not image_paths:
        raise ValueError("'image_paths' must be a non-empty list.")

    logger_setup(log_file=config.log_file)
    trainer = TrainerCNN(config, config.data_root, checkpoint_dir=config.checkpoint_dir)
    trainer.load_checkpoint(checkpoint_path)

    device = next(trainer.model.parameters()).device
    transform = get_val_transforms(config.image_size)
    images = torch.stack(
        [transform(Image.open(p).convert("RGB")) for p in image_paths]
    ).to(device)

    generated_ids = trainer.model.generate(
        images,
        sos_token_id=trainer.tokenizer.vocab.sos_idx,
        eos_token_id=trainer.tokenizer.vocab.eos_idx,
        max_len=max_len,
    )
    captions = [
        trainer.tokenizer.vocab.decode(row.tolist(), skip_special=True)
        for row in generated_ids
    ]
    for path, caption in zip(image_paths, captions):
        print(f"{Path(path).name}: {caption}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image captioning — train or generate captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "architecture",
        choices=["vit", "cnn"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "mode",
        choices=["train", "gen"],
        help="Whether to train or generate captions.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a JSON config file. Omit to use default values.",
    )
    return parser.parse_args()


def main() -> None:
    logger_setup()
    args = parse_args()
    cfg_dict = load_json_config(args.config)

    dispatch = {
        ("vit", "train"): run_train_vit,
        ("vit", "gen"): run_gen_vit,
        ("cnn", "train"): run_train_cnn,
        ("cnn", "gen"): run_gen_cnn,
    }
    dispatch[(args.architecture, args.mode)](cfg_dict)


if __name__ == "__main__":
    main()
