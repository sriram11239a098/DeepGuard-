# DeepGuard / Sach-AI — Complete Setup Guide
## Python 3.11 · Windows 11 · NVIDIA RTX 3050 6 GB

---

## 1 · Project Folder Structure

```
DeepGuard/
├── config.py                  ← EDIT dataset paths here
├── main.py                    ← Single CLI entrypoint
├── train.py                   ← Training script
├── app.py                     ← Streamlit web UI
├── api.py                     ← FastAPI REST server
├── arrange_datasets.py        ← One-time dataset organiser
├── install.py                 ← One-shot dependency installer
├── download_torch.py          ← Resumable PyTorch wheel downloader
├── requirements.txt
├── launch.bat                 ← Windows quick-launch menu
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py       ← Frame & audio extraction (Module 1)
│   ├── transforms.py          ← Albumentations transforms
│   ├── image_model.py         ← EfficientNet-B0 image branch (Module 4)
│   ├── audio_model.py         ← ResNet-18 on Mel-spec branch (Module 3)
│   ├── video_model.py         ← EfficientNet-B0 + BiGRU video branch (Module 2)
│   ├── fusion_model.py        ← Unified multimodal fusion model (Module 5)
│   ├── fusion.py              ← Legacy weighted-average fusion utility
│   └── dataset.py             ← PyTorch Dataset class
│
├── utils/
│   ├── __init__.py
│   ├── inference.py           ← DeepGuardInference engine
│   ├── logger.py              ← Logging setup
│   ├── metrics.py             ← AUC, accuracy, F1 etc.
│   └── tictoc.py              ← Wall-clock timer
│
├── checkpoints/               ← Auto-created; .pth weights saved here
├── outputs/                   ← Auto-created; reports saved here
└── logs/                      ← Auto-created; training logs saved here
```

---

## 2 · Dataset Folder Layout

```
E:\CIP\Datasets_Ready\
├── Audio\
│   ├── real\       ← genuine audio  (.wav / .flac / .mp3)
│   └── fake\       ← spoofed audio
│
├── Image\
│   ├── real\       ← genuine face images (.jpg / .png)
│   └── fake\       ← GAN / manipulated images
│
└── Video\
    ├── real\       ← genuine videos (.mp4 / .avi)
    └── fake\       ← deepfake videos
```

> If your raw dataset does not follow this layout, run:
> `python arrange_datasets.py`   (edit SOURCE_DIR / TARGET_DIR first)

---

## 3 · Installation

### 3.1 — Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 3.2 — Install PyTorch with CUDA 12.1 (RTX 3050 compatible)

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ✅ Do this **before** step 3.3 — pip resolves PyTorch separately.

### 3.3 — Install remaining dependencies

```bat
pip install -r requirements.txt
```

### 3.4 — Install ffmpeg (required for audio extraction from video)

1. Download from https://ffmpeg.org/download.html  
2. Extract and add the `bin/` folder to your PATH.  
3. Verify: `ffmpeg -version`

Without ffmpeg, the audio branch returns a zero spectrogram for video inputs
(training and inference still work; audio branch score will be unreliable).

---

## 4 · Verify the Installation

```bat
python main.py check
```

Expected output:
```
── Environment Check ─────────────────────────────────────
  Python      : 3.11.x
  PyTorch     : 2.x.x+cu121
  CUDA avail  : True
  GPU         : NVIDIA GeForce RTX 3050 Laptop GPU
  Device used : cuda

── Dataset Paths ─────────────────────────────────────────
  Video  ✓  →  E:\CIP\Datasets_Ready\Video
          real: 250 files   fake: 250 files
  Audio  ✓  →  E:\CIP\Datasets_Ready\Audio
          real: 250 files   fake: 250 files
  Image  ✓  →  E:\CIP\Datasets_Ready\Image
          real: 250 files   fake: 250 files

── Checkpoints ───────────────────────────────────────────
  latest_fusion_model: ✗ not trained yet
  best_fusion_model  : ✗ not trained yet
```

---

## 5 · Training

```bat
python main.py train
```

Checkpoints are saved to `checkpoints/` after every epoch and whenever
validation AUC improves.  Training auto-resumes from `latest_fusion_model.pth`
if interrupted.

Key config options (edit `config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `BATCH_SIZE` | 4 | Reduce to 2 if CUDA OOM |
| `GRADIENT_ACCUMULATION_STEPS` | 8 | Effective batch = BATCH_SIZE × 8 |
| `NUM_EPOCHS` | 20 | Increase for better accuracy |
| `LEARNING_RATE` | 1e-4 | AdamW learning rate |
| `PATIENCE` | 5 | Early stopping patience |
| `VIDEO_MAX_FRAMES` | 8 | Frames sampled per video |
| `DETECTION_THRESHOLD` | 0.5 | P(fake) ≥ threshold → DEEPFAKE |
| `NUM_WORKERS` | 0 | Keep 0 on Windows |

---

## 6 · Inference (Single File)

```bat
python main.py predict --file "E:\CIP\test_samples\sample.mp4"
python main.py predict --file "E:\CIP\test_samples\sample.wav"
python main.py predict --file "E:\CIP\test_samples\sample.jpg"
```

Sample output:
```
==================================================
  VERDICT     : DEEPFAKE
  CONFIDENCE  : 87.3%
  Video score : 0.873
  Audio score : 0.873
  Image score : 0.873
==================================================
```

---

## 7 · Web Dashboard (Streamlit)

```bat
python main.py ui
```

Open: http://localhost:8501

---

## 8 · REST API (FastAPI)

```bat
python main.py api
```

Swagger UI: http://localhost:8000/docs

PowerShell example:
```powershell
curl -X POST http://localhost:8000/detect -F "file=@C:\path\to\video.mp4"
```

---

## 9 · Troubleshooting

| Problem | Fix |
|---|---|
| `torch not installed` | Run Step 3.2 before requirements.txt |
| `CUDA not available` | Check driver: `nvidia-smi` in CMD |
| `No files found in dataset` | Check `real/` and `fake/` subfolders exist |
| `OOM / CUDA out of memory` | Lower `BATCH_SIZE` to 2 in config.py |
| `ffmpeg not found` | Install ffmpeg and add to PATH |
| `JpegCompression not found` | `pip install albumentations>=1.4.0` |
| `AttributeError: config['data']` | Fixed — config is now used as a module |
| Slow training | Set `NUM_WORKERS=2` on Linux; keep 0 on Windows |
