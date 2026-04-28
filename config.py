"""
DeepGuard / Sach-AI — Central Configuration File
Modify the DATASET paths below to match your local setup.
All other settings can be tuned here without touching module code.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── Local Binaries (ffmpeg, etc) ─────────────────────────────────────────────
LOCAL_BIN_DIR = PROJECT_ROOT / "bin"
if LOCAL_BIN_DIR.exists():
    # Prepend local bin to PATH for the current process
    os.environ["PATH"] = str(LOCAL_BIN_DIR) + os.pathsep + os.environ["PATH"]


# ─────────────────────────────────────────────
# DATASET PATHS  ← EDIT THESE
# ─────────────────────────────────────────────
DATASET_ROOT_DIR  = Path(r"E:\CIP\Datasets_Ready")

# Specific Modality Folders
DATASET_AUDIO_DIR = DATASET_ROOT_DIR / "Audio"
DATASET_IMAGE_DIR = DATASET_ROOT_DIR / "Image"
DATASET_VIDEO_DIR = DATASET_ROOT_DIR / "Video"

# Expected subdirectory layout inside each dataset folder:
#   Audio/
#     real/   ← genuine audio files (.wav / .flac / .mp3)
#     fake/   ← spoofed audio files
#   Image/
#     real/   ← genuine face images (.jpg / .png)
#     fake/   ← GAN-generated / manipulated images
#   Video/
#     real/   ← genuine video files (.mp4 / .avi)
#     fake/   ← deepfake video files

# ─────────────────────────────────────────────
# OUTPUT / CHECKPOINT PATHS
# ─────────────────────────────────────────────
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
LOGS_DIR        = PROJECT_ROOT / "logs"

for _d in [CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
# We use a simple string here to avoid importing torch at startup.
# The actual check is deferred or handled by the modules that need it.
DEVICE = "cuda" # Assume cuda, modules will fallback if needed

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
IMAGE_SIZE       = 160          # Reduced from 224 for faster training
FACE_MARGIN      = 20           # Pixels to add around detected face bbox
AUDIO_SR         = 16000        # Target sample rate (Hz)
AUDIO_DURATION   = 3            # Seconds per audio window
AUDIO_HOP        = 1            # Hop between windows (seconds)
N_MELS           = 128          # Mel spectrogram bins
VIDEO_MAX_FRAMES = 4            # Reduced from 8 for faster training

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────
FUSION_HIDDEN_DIM   = 256
IMAGE_BACKBONE      = "densenet121"
AUDIO_BACKBONE      = "custom_cnn"
VIDEO_BACKBONE      = "resnext50_32x4d"

# ─────────────────────────────────────────────
# FUSION WEIGHTS
# These are used as fallback in the legacy fuse() utility.
# The trained MultimodalFusionModel learns its own weights.
# ─────────────────────────────────────────────
FUSION_WEIGHT_VIDEO = 0.40
FUSION_WEIGHT_AUDIO = 0.30
FUSION_WEIGHT_IMAGE = 0.30

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
BATCH_SIZE                  = 12   # Increased from 8 for faster training
GRADIENT_ACCUMULATION_STEPS = 2    # Reduced from 4 to maintain effective batch size (~24-32)
MIXED_PRECISION             = True # FP16 — set False on CPU
NUM_EPOCHS                  = 20
LEARNING_RATE               = 1e-4
WEIGHT_DECAY                = 1e-5
VAL_SPLIT                   = 0.1
NUM_WORKERS                 = 4    # Increased from 0 to enable multi-process pre-fetching
PATIENCE                    = 5

# ─────────────────────────────────────────────
# INFERENCE THRESHOLD
# ─────────────────────────────────────────────
DETECTION_THRESHOLD = 0.5      # P(fake) >= threshold → DEEPFAKE

# ─────────────────────────────────────────────
# API / UI
# ─────────────────────────────────────────────
LOOPBACK_ADDR = "127.0.0.1"
API_HOST      = LOOPBACK_ADDR
API_PORT      = 8000
UI_PORT       = 8501
