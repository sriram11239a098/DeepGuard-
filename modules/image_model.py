"""
image_model.py — Image branch: DenseNet121 backbone → embedding.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class ImageModel(nn.Module):
    """
    DenseNet121 feature extractor for single images.

    Input  : (B, 3, H, W)
    Output : (B, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)

        # Replace the final classifier with Identity to get raw features
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features   = self.backbone(x)    # (B, in_features)
        embeddings = self.fc(features)   # (B, hidden_dim)
        return embeddings
