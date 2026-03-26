import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class CNNEncoder(nn.Module):
    def __init__(
            self, 
            embed_size: int = 512, 
            pretrained: bool = True, 
            freeze: bool = True
    ):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # Remove the last layer(classification layer)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        
        # Linear layer to project visual features to the embedding space
        self.proj = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        
        # Option to freeze the encoder weights
        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad = False
                
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, 224, 224]
        returns: [B, 1, embed_size]
        """
        features = self.resnet(images) # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1) # Flatten to [B, 2048]
        features = self.proj(features) # [B, embed_size]
        # We unsqueeze to [B, 1, D] to mimic the "memory" interface of the ViT
        return self.relu(features).unsqueeze(1)