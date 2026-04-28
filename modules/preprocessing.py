"""
preprocessing.py — Media preprocessing utilities for DeepGuard / Sach-AI.

Provides:
    extract_frames(video_path, max_frames)  → list of RGB numpy arrays
    extract_audio_mel(path, ...)            → torch.Tensor (1, n_mels, T)
"""

import os
import subprocess
import tempfile

import cv2
import numpy as np
import torch
import torchaudio


# ─────────────────────────────────────────────────────────────────────────────
# Video: Frame Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, max_frames: int = 8) -> list:
    """
    Read a video and return `max_frames` evenly-spaced RGB frames.

    Returns a list of numpy arrays with shape (H, W, 3), dtype uint8.
    On failure returns a list of zero-filled frames.
    """
    blank = np.zeros((224, 224, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video: {video_path}")
        return [blank.copy() for _ in range(max_frames)]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback or very long videos: read sequentially with safety limit
    if total_frames <= 0 or total_frames > 5000:
        if total_frames > 5000:
            print(f"[WARNING] Video too long ({total_frames} frames). Limiting search.")
        
        frames_list = []
        # Read a maximum of 2000 frames to sample from, to avoid OOM
        for _ in range(2000):
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        cap.release()
        
        count = len(frames_list)
        if count == 0:
            return [blank.copy() for _ in range(max_frames)]
            
        indices = np.linspace(0, count - 1, max_frames, dtype=int)
        extracted = []
        for idx in indices:
            frame = cv2.cvtColor(frames_list[idx], cv2.COLOR_BGR2RGB)
            extracted.append(frame)
        while len(extracted) < max_frames:
            extracted.append(blank.copy())
        return extracted

    # Standard case: Sequential read with skipping (faster and more reliable than set/seek)
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    extracted = []
    
    current_idx = 0
    for target_idx in indices:
        # Skip frames sequentially to reach target
        while current_idx < target_idx:
            cap.grab()
            current_idx += 1
        
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted.append(frame)
        else:
            extracted.append(blank.copy())
        current_idx += 1

    cap.release()
    while len(extracted) < max_frames:
        extracted.append(blank.copy())

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Audio: Mel-Spectrogram Extraction
# ─────────────────────────────────────────────────────────────────────────────

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def extract_audio_mel(
    video_or_audio_path: str,
    sample_rate: int  = 16000,
    duration:    int  = 3,
    n_mels:      int  = 128,
) -> torch.Tensor:
    """
    Extract audio from a video or read an audio file and return a
    log-Mel spectrogram tensor of shape (1, n_mels, T).

    For video files, ffmpeg is used to extract audio.  If ffmpeg is
    unavailable or extraction fails, a zero tensor is returned so that
    training / inference can continue without crashing.
    """
    ext = os.path.splitext(video_or_audio_path)[1].lower()
    waveform = None
    sr       = sample_rate

    try:
        if ext in _AUDIO_EXTS:
            try:
                waveform, sr = torchaudio.load(video_or_audio_path)
            except Exception as e:
                # Fallback to librosa for audio files (more robust but slower)
                import librosa
                y, sr = librosa.load(video_or_audio_path, sr=sample_rate)
                waveform = torch.from_numpy(y).unsqueeze(0)
        else:
            # Extract audio track from video using ffmpeg
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            try:
                # Check if ffmpeg exists first to avoid confusing WinError 2
                subprocess.run(
                    ["ffmpeg", "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i",      video_or_audio_path,
                        "-vn",
                        "-acodec", "pcm_s16le",
                        "-ar",     str(sample_rate),
                        "-ac",     "1",
                        "-loglevel", "quiet",
                        tmp_path,
                    ],
                    check=True,
                    timeout=30,
                )
                waveform, sr = torchaudio.load(tmp_path)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # We already handle waveform is None later
                pass
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as e:
        # Final catch-all to prevent training crashes
        pass

    if waveform is None:
        # Return silence — downstream model still gets a valid tensor
        time_steps = int(sample_rate * duration / 512) + 1
        return torch.zeros(1, n_mels, time_steps)

    # ── Resample if needed ────────────────────────────────────────────────────
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=sample_rate
        )
        waveform = resampler(waveform)

    # ── Convert to mono ───────────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # ── Pad / trim to target duration ─────────────────────────────────────────
    target_length = sample_rate * duration
    if waveform.shape[1] >= target_length:
        waveform = waveform[:, :target_length]
    else:
        pad = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))

    # ── Mel Spectrogram ───────────────────────────────────────────────────────
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512,
    )
    mel_spec = mel_transform(waveform)                         # (1, n_mels, T)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec) # log scale

    # ── Normalisation ─────────────────────────────────────────────────────────
    # Standardize to zero mean and unit variance for better model convergence
    eps = 1e-6
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + eps)

    return mel_spec
