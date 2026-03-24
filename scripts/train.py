import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from Image_Captioning_Architectures.data.flickr8k_dataset import (
    build_flickr8k_dataloaders,
)
from Image_Captioning_Architectures.models.decoder import CaptionDecoder
from Image_Captioning_Architectures.models.model import ImageCaptioningModel
from Image_Captioning_Architectures.models.vit_encoder import ViTEncoder

# Allow running from project root:
# python scripts/train.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_model(
    vocab_size: int,
    pad_token_id: int,
    freeze_encoder: bool = True,
) -> ImageCaptioningModel:
    encoder = ViTEncoder(
        model_name="vit_base_patch16_224",
        pretrained=True,
        decoder_dim=512,
        freeze=freeze_encoder,
    )

    decoder = CaptionDecoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=128,
        pad_token_id=pad_token_id,
    )

    model = ImageCaptioningModel(
        encoder=encoder,
        decoder=decoder,
        pad_token_id=pad_token_id,
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)

        optimizer.zero_grad()
        loss, logits = model.compute_loss(images, captions, criterion)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"batch={batch_idx} loss={loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)

        loss, logits = model.compute_loss(images, captions, criterion)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = os.path.join(PROJECT_ROOT, "data", "datasets", "flickr8k")
    batch_size = 8
    num_workers = 0
    learning_rate = 1e-4
    num_epochs = 2

    train_loader, val_loader, test_loader, vocab = build_flickr8k_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=224,
        min_freq=1,
        num_workers=num_workers,
    )

    print("vocab size:", len(vocab))
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    model = build_model(
        vocab_size=len(vocab),
        pad_token_id=vocab.pad_id,
        freeze_encoder=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"epoch={epoch + 1}/{num_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

    batch = next(iter(val_loader))
    sample_image = batch["images"][0].to(device)
    sample_caption = batch["raw_captions"][0]

    generated_ids = model.generate(
        images=sample_image,
        sos_token_id=vocab.sos_id,
        eos_token_id=vocab.eos_id,
        max_len=20,
    )[0].tolist()

    generated_tokens = [vocab.itos[idx] for idx in generated_ids if idx in vocab.itos]
    print("reference:", sample_caption)
    print("generated:", " ".join(generated_tokens))


if __name__ == "__main__":
    main()
