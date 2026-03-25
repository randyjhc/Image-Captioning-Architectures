"""Entry point for ViT image captioning training.

Run with:
    python train_vit/train.py
    python -m train_vit.train
"""

from __future__ import annotations

from pathlib import Path

from model_vit import GeneratorViT
from train_vit.config import ConfigViT
from train_vit.trainer import TrainerViT


def main() -> None:
    config = ConfigViT()
    data_root = Path(__file__).parent.parent / "data" / "datasets" / "flickr8k"

    trainer = TrainerViT(
        config=config,
        data_root=data_root,
        checkpoint_dir=Path("checkpoints/vit"),
    )

    print(f"vocab size: {len(trainer.tokenizer.vocab)}")
    print(f"device: {trainer._device}")

    trainer.fit()

    # Sample generation after training
    val_image, _ = trainer.val_dataset[0]
    device = next(trainer.model.parameters()).device
    generator = GeneratorViT(trainer.model, trainer.tokenizer)
    captions = generator.generate_caption(val_image.unsqueeze(0).to(device), max_len=20)

    print("reference:", trainer.val_dataset.get_caption(0))
    print("generated:", captions[0])


if __name__ == "__main__":
    main()
