import torch
import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, pad_token_id: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_token_id = pad_token_id

    def forward(self, images, input_ids):
        # Extract features and predict words
        features = self.encoder(images) # [B, 1, D]
        logits = self.decoder(input_ids, features) # [B, T, vocab_size]
        return logits

    def compute_loss(self, images, captions, criterion):
        # Teacher forcing: feed 0 to T-1, predict 1 to T
        input_ids = captions[:, :-1]
        targets = captions[:, 1:]
        
        logits = self.forward(images, input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, logits

    @torch.no_grad()
    def generate(self, images, sos_token_id, eos_token_id, max_len=30):
        # Greedy search logic for evaluate_cnn.py [cite: 63]
        self.eval()
        features = self.encoder(images)
        generated = torch.full((images.size(0), 1), sos_token_id, dtype=torch.long, device=images.device)
        
        for _ in range(max_len - 1):
            logits = self.decoder(generated, features)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token == eos_token_id): break
        return generated