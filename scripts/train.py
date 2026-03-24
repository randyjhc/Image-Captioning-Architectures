import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.flickr_dataset import FlickrDataset
from data.image.transforms import (
    get_train_transforms,
    get_val_transforms,
)
from data.text.vocabulary import (
    CaptionTokenizer,
    Vocabulary,
)
from models.decoder import CaptionDecoder
from models.model import ImageCaptioningModel
from models.vit_encoder import ViTEncoder


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

    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)

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

    for images, captions in dataloader:
        images = images.to(device)
        captions = captions.to(device)

        loss, logits = model.compute_loss(images, captions, criterion)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path(__file__).parent.parent / "data" / "datasets" / "flickr8k"
    batch_size = 8
    num_workers = 0
    learning_rate = 1e-4
    num_epochs = 2

    captions_file = Path(data_root) / "captions.txt"
    with open(captions_file, encoding="utf-8") as f:
        all_captions = [row["caption"] for row in csv.DictReader(f)]

    vocab = Vocabulary.build_from_captions(all_captions, min_freq=1)
    tokenizer = CaptionTokenizer(vocab)

    train_ds, val_ds, test_ds = FlickrDataset.create_splits(
        root_dir=data_root,
        train_transform=get_train_transforms(224),
        val_transform=get_val_transforms(224),
        test_transform=get_val_transforms(224),
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("vocab size:", len(vocab))
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    model = build_model(
        vocab_size=len(vocab),
        pad_token_id=vocab.pad_idx,
        freeze_encoder=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"epoch={epoch + 1}/{num_epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

    sample_image, _ = val_ds[0]
    sample_caption = val_ds.get_caption(0)

    generated_ids = model.generate(
        images=sample_image.to(device),
        sos_token_id=vocab.sos_idx,
        eos_token_id=vocab.eos_idx,
        max_len=20,
    )[0].tolist()

    generated_tokens = [
        vocab.idx2word[idx] for idx in generated_ids if idx in vocab.idx2word
    ]
    print("reference:", sample_caption)
    print("generated:", " ".join(generated_tokens))


if __name__ == "__main__":
    main()
