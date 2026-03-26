import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.flickr_dataset import FlickrDataset
from data.image.transforms import get_train_transforms, get_val_transforms
from data.text.vocabulary import CaptionTokenizer, Vocabulary

from model_cnn_lstm import CNNEncoder, LSTMDecoder, ImageCaptioningModel


def build_model(vocab_size, pad_token_id):
    encoder = CNNEncoder(embed_size=512, pretrained=True, freeze=True)
    decoder = LSTMDecoder(
        vocab_size=vocab_size, 
        embed_size=512, 
        hidden_size=512, 
        pad_token_id=pad_token_id
    )
    
    return ImageCaptioningModel(encoder, decoder, pad_token_id=pad_token_id)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, captions) in enumerate(tqdm(dataloader, desc="Training")):
        images, captions = images.to(device), captions.to(device)

        optimizer.zero_grad()
        loss, _ = model.compute_loss(images, captions, criterion)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for images, captions in tqdm(dataloader, desc="Validating"):
        images, captions = images.to(device), captions.to(device)
        loss, _ = model.compute_loss(images, captions, criterion)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "datasets" / "flickr8k"
    
    batch_size = 32 
    learning_rate = 1e-4
    num_epochs = 10
    
    captions_file = data_root / "captions.txt"
    with open(captions_file, encoding="utf-8") as f:
        all_captions = [row["caption"] for row in csv.DictReader(f)]
    
    vocab = Vocabulary.build_from_captions(all_captions, min_freq=1)
    tokenizer = CaptionTokenizer(vocab)

    train_ds, val_ds, _ = FlickrDataset.create_splits(
        root_dir=data_root,
        train_transform=get_train_transforms(224),
        val_transform=get_val_transforms(224),
        test_transform=get_val_transforms(224),
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = build_model(len(vocab), vocab.pad_idx).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    save_path = project_root / "cnn_lstm_baseline.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()