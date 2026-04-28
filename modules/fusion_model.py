"""
fusion_model.py — Multimodal Fusion Model for DeepGuard / Sach-AI.

Concatenates embeddings from all three branches and runs a small MLP
to produce a single binary logit (real vs. fake).
"""

import torch
import torch.nn as nn

from .image_model import ImageModel
from .audio_model import AudioModel
from .video_model import VideoModel


class MultimodalFusionModel(nn.Module):
    """
    Unified model that jointly processes image, audio, and video inputs.

    Forward args:
        x_img   : (B, 3,        H,    W)
        x_audio : (B, 1,   n_mels,    T)
        x_video : (B, T,        3, H, W)

    Returns:
        logits  : (B, 1)   — raw (pre-sigmoid) output; use BCEWithLogitsLoss
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.image_model = ImageModel(hidden_dim=hidden_dim)
        self.audio_model = AudioModel(hidden_dim=hidden_dim)
        self.video_model = VideoModel(hidden_dim=hidden_dim)

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1),   # binary logit
        )

    def forward(
        self,
        x_img:   torch.Tensor,
        x_audio: torch.Tensor,
        x_video: torch.Tensor,
    ) -> torch.Tensor:
        img_emb   = self.image_model(x_img)    # (B, hidden_dim)
        audio_emb = self.audio_model(x_audio)  # (B, hidden_dim)
        video_emb = self.video_model(x_video)  # (B, hidden_dim)

        fused  = torch.cat([img_emb, audio_emb, video_emb], dim=1)  # (B, hidden_dim*3)
        logits = self.fusion_fc(fused)                               # (B, 1)
        return logits
