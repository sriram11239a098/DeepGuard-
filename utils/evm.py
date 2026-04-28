"""
evm.py — Simplified Eulerian Video Magnification (EVM) for DeepGuard.

Amplifies temporal color variations in a frame sequence to expose
GAN-generated artefacts.
"""

import numpy as np
import cv2


def apply_evm(frames: list[np.ndarray], alpha: float = 20.0, low_f: float = 0.4, high_f: float = 3.0) -> list[np.ndarray]:
    """
    Apply simplified EVM to a list of frames.
    
    1. Convert frames to float32.
    2. Apply a temporal bandpass filter (approximated by differences).
    3. Amplify the filtered signal by alpha.
    4. Add back to original frames and clip.
    """
    if not frames:
        return []

    # Convert to float and stack
    video = np.array(frames, dtype=np.float32) / 255.0
    t, h, w, c = video.shape

    if t < 3:
        return frames

    # Temporal bandpass approximation: Simple difference or Gaussian blur in time
    # Here we use a simpler approach: amplify high-frequency temporal changes
    
    # Calculate mean across time
    mean_frame = np.mean(video, axis=0)
    
    # Magnified video: original + alpha * (frame - mean)
    # This amplifies deviations from the average color over time
    magnified = video + alpha * (video - mean_frame)
    
    # Clip and convert back to uint8
    magnified = np.clip(magnified * 255.0, 0, 255).astype(np.uint8)
    
    return [magnified[i] for i in range(t)]


def get_evm_diff_map(frames: list[np.ndarray]) -> np.ndarray:
    """
    Returns a single 'heat map' of temporal activity.
    Useful for visualizing where EVM found the most change.
    """
    if not frames:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    video = np.array(frames, dtype=np.float32)
    diff  = np.abs(np.diff(video, axis=0))
    mean_diff = np.mean(diff, axis=0)
    
    # Normalize to 0-255
    res = cv2.normalize(mean_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(res[:, :, 0], cv2.COLORMAP_JET)
