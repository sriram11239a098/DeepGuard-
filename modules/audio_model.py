"""
audio_model.py — Audio branch: Custom CNN on Mel spectrograms → embedding.
"""

import torch
import torch.nn as nn


class AudioModel(nn.Module):
    """
    Custom CNN stack that accepts single-channel Mel spectrograms.
    Architecture: Conv → MaxPool → BatchNorm → ReLU → Dropout → GAP → Dense.

    Input  : (B, 1, n_mels, T)
    Output : (B, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        feats      = self.features(x)   # (B, 256, 1, 1)
        embeddings = self.fc(feats)     # (B, hidden_dim)
        return embeddings
