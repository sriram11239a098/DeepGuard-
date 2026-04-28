"""
transforms.py — Albumentations image/video transforms for DeepGuard.
Handles both training (with augmentation) and inference (clean resize + normalise).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_image_transforms(image_size: int = 224, is_train: bool = True) -> A.Compose:
    """
    Returns albumentations transforms for single-image processing.

    Training augmentations:
        - HorizontalFlip, RandomBrightnessContrast, GaussianBlur
        - JPEG compression artefacts (robustness against compression deepfakes)
        - ImageNet normalisation → tensor

    Inference:
        - Resize + normalise → tensor only (no stochastic ops)
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            # ImageCompression is the correct name in albumentations ≥ 2.0
            A.ImageCompression(quality_range=(60, 100), p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ])


def get_video_transforms(image_size: int = 224, is_train: bool = True) -> A.Compose:
    """
    Per-frame transforms for video branches.
    Reuses get_image_transforms — keeps spatial consistency across frames.
    """
    return get_image_transforms(image_size, is_train)
