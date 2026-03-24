import torch.nn as nn
import timm


class ViTEncoder(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        pretrained=True,
        decoder_dim=512,
        freeze=False,
    ):
        super().__init__()

        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        vit_dim = self.vit.num_features
        self.proj = nn.Linear(vit_dim, decoder_dim)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, images):
        # forward_features usually returns token sequence for ViT in timm
        x = self.vit.forward_features(images)

        # Depending on model, x may be [B, N, D] or a dict / special format.
        # For standard ViT in timm, usually [B, N, D].
        if isinstance(x, dict):
            x = x["x"]

        # remove CLS token if present: keep patch tokens for cross-attention
        if x.dim() == 3 and x.size(1) > 1:
            x = x[:, 1:, :]  # [B, num_patches, vit_dim]

        x = self.proj(x)  # [B, num_patches, decoder_dim]
        return x
