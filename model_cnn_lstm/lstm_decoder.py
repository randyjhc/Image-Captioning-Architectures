import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int = 512, 
        hidden_size: int = 512, 
        num_layers: int = 1,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids) # [B, T, embed_size]
        
        # Concatenate the image feature to the front of the caption embeddings
        combined = torch.cat((memory, embeddings), dim=1) # [B, T+1, embed_size]
        
        hiddens, _ = self.lstm(combined) # [B, T+1, hidden_size]
        
        # Predict next words (dropping the last prediction and shifting)
        outputs = self.linear(hiddens[:, 1:, :]) # [B, T, vocab_size]
        return outputs