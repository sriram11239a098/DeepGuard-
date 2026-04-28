"""
video_model.py — Video branch: ResNext50 per-frame + LSTM → embedding.
"""

import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class VideoModel(nn.Module):
    """
    Per-frame ResNext50 feature extractor followed by a
    Stacked LSTM for temporal aggregation.

    Input  : (B, T, 3, H, W)   — batch of T-frame clips
    Output : (B, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim:      int  = 256,
        pretrained:      bool = True,
        lstm_hidden_size: int  = 128,
        lstm_layers:      int  = 2,
    ):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        self.backbone = resnext50_32x4d(weights=weights)

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False, # Per doc "Stacked LSTM" usually implies uni-directional
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # Extract per-frame spatial features
        x_flat         = x.view(b * t, c, h, w)
        spatial_feats  = self.backbone(x_flat)          # (B*T, feature_dim)
        spatial_feats  = spatial_feats.view(b, t, -1)  # (B, T, feature_dim)

        # Temporal aggregation
        lstm_out, _ = self.lstm(spatial_feats)          # (B, T, lstm_hidden)
        last_out    = lstm_out[:, -1, :]                # (B, lstm_hidden)

        embeddings = self.fc(last_out)                  # (B, hidden_dim)
        return embeddings
