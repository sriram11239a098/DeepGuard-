"""
dataset.py — Unified multimodal dataset for DeepGuard / Sach-AI.

Handles images, videos, and audio files under a real/fake directory layout.
Uses lazy loading to avoid exhausting RAM on large datasets.
"""

import cv2
import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset

import config as cfg
from .preprocessing import extract_frames, extract_audio_mel
from .transforms import get_image_transforms, get_video_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Supported extensions per modality
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Keywords in path components that indicate a real (label=0) sample
REAL_KEYWORDS = {"real", "authentic", "original"}


class MultimodalDeepfakeDataset(Dataset):
    """
    Scans `data_dir` recursively for media files inside `real/` and `fake/`
    subdirectories, then splits them 80/10/10 (train / val / test) in a
    deterministic, sorted order so splits are always identical.

    Args:
        data_dir : root directory — may contain Audio/, Image/, Video/ sub-
                   folders, each with real/ and fake/ children, or a flat
                   real/ + fake/ layout.
        split    : "train" | "val" | "test"
    """

    def __init__(self, data_dir, split: str = "train"):
        self.split    = split
        self.is_train = split == "train"

        self.image_transform = get_image_transforms(cfg.IMAGE_SIZE, self.is_train)
        self.video_transform = get_video_transforms(cfg.IMAGE_SIZE, self.is_train)

        self.data = self._scan_and_split(Path(data_dir), split)

        if len(self.data) == 0:
            print(
                f"[WARNING] No valid media files found in '{data_dir}' "
                f"for split='{split}'."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _scan_and_split(self, root: Path, split: str) -> pd.DataFrame:
        if not root.exists():
            return pd.DataFrame(columns=["file_path", "label", "modality"])

        all_exts  = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS
        all_files = sorted(
            f for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in all_exts
        )

        # CRITICAL FIX: Shuffle files before splitting to avoid class imbalance
        # (e.g. all 'real' in train, all 'fake' in val)
        random.seed(42)
        random.shuffle(all_files)

        total    = len(all_files)
        t_end    = int(total * 0.8)
        v_end    = t_end + int(total * 0.1)

        if   split == "train": files = all_files[:t_end]
        elif split == "val":   files = all_files[t_end:v_end]
        else:                  files = all_files[v_end:]

        records = []
        for fp in files:
            ext = fp.suffix.lower()
            if   ext in IMAGE_EXTS: modality = "image"
            elif ext in VIDEO_EXTS: modality = "video"
            elif ext in AUDIO_EXTS: modality = "audio"
            else: continue

            parts  = {p.lower() for p in fp.parts}
            label  = 0 if parts & REAL_KEYWORDS else 1   # 0=real, 1=fake
            records.append({"file_path": str(fp), "label": label, "modality": modality})

        return pd.DataFrame(records)

    # ── Expected audio spectrogram width ─────────────────────────────────────
    @property
    def _audio_time_steps(self) -> int:
        return int(cfg.AUDIO_SR * cfg.AUDIO_DURATION / 512) + 1

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row      = self.data.iloc[idx]
        fp       = row["file_path"]
        label    = int(row["label"])
        modality = str(row["modality"])

        # Default zero tensors for all modalities
        image_tensor = torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        video_tensor = torch.zeros(
            cfg.VIDEO_MAX_FRAMES, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE
        )
        audio_tensor = torch.zeros(1, cfg.N_MELS, self._audio_time_steps)

        try:
            if modality == "video":
                frames = extract_frames(fp, cfg.VIDEO_MAX_FRAMES)

                # Video tensor — all frames
                t_frames = [self.video_transform(image=f)["image"] for f in frames]
                video_tensor = torch.stack(t_frames)             # (T, 3, H, W)

                # Image tensor — middle frame
                mid_frame  = frames[len(frames) // 2]
                image_tensor = self.image_transform(image=mid_frame)["image"]

                # Audio tensor
                audio_tensor = extract_audio_mel(
                    fp, cfg.AUDIO_SR, cfg.AUDIO_DURATION, cfg.N_MELS
                )

            elif modality == "image":
                img = cv2.imread(fp)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_tensor = self.image_transform(image=img)["image"]

            elif modality == "audio":
                audio_tensor = extract_audio_mel(
                    fp, cfg.AUDIO_SR, cfg.AUDIO_DURATION, cfg.N_MELS
                )

        except Exception as e:
            # Keep zero tensors so a single corrupt file never halts training
            print(f"[WARNING] Failed to load '{fp}': {e}")

        return {
            "image": image_tensor,
            "video": video_tensor,
            "audio": audio_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
        }
