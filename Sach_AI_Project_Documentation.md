# 🛡️ Sach-AI: Multimodal Deepfake Detection System

> **Project Title:** Fake Trace — Deep Learning-Based Deepfake Detection System for Digital Forensic Investigation
> **Institution:** Sri Chandrasekharendra Saraswathi Viswa Mahavidyalaya (SCSVMV University)
> **Department:** Computer Science & Engineering
> **Duration:** January 2026 – May 2026
> **Domain:** Deep Learning | Computer Vision | Digital Forensics | Generative AI

---

## 📋 Table of Contents

1. [Abstract](#1-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Existing Systems](#3-existing-systems)
4. [Proposed System — Sach-AI](#4-proposed-system--sach-ai)
5. [System Architecture](#5-system-architecture)
6. [Module Descriptions](#6-module-descriptions)
7. [End-to-End Workflow](#7-end-to-end-workflow)
8. [System Requirements](#8-system-requirements)
9. [Technology Stack](#9-technology-stack)
10. [Conclusion](#10-conclusion)

---

## 1. Abstract

The rapid advancement of **Generative Adversarial Networks (GANs)** and diffusion-based AI models has made it alarmingly easy to create highly realistic synthetic media — commonly known as **deepfakes**. These AI-generated artefacts, where a person's face or voice is convincingly manipulated, pose severe threats to:

- **Digital forensics** and law enforcement investigations
- **Journalism integrity** and factual reporting
- **National security** and public trust

Existing detection frameworks are limited to a single modality (video, audio, or image), rendering them inadequate against modern multimodal attacks. This project proposes **Sach-AI** — a unified, real-time deepfake detection framework grounded in the research paper by Sar et al. (IEEE Access, 2025) — that simultaneously analyses video, audio, and image content using three specialised deep learning pipelines fused under a single architecture.

The system delivers a **forensic confidence report** complete with Grad-CAM heatmaps, per-modality verdicts, and an explainable authenticity score — making it a deployable tool for digital evidence analysis in investigative contexts.

---

## 2. Problem Statement

### 2.1 Core Challenges

| Challenge | Description |
|---|---|
| **Single-Modality Limitation** | Current detection systems address only one medium (video, audio, or image) in isolation, leaving them blind to cross-modal attacks. |
| **Accuracy Gap** | State-of-the-art tools such as Intel's FakeCatcher achieve ~96% accuracy on video deepfakes but provide zero coverage for audio or image manipulation. |
| **Dataset Narrow-ness** | Most prior models over-fit to specific training sets and generalise poorly to unseen deepfake generation algorithms. |
| **Scalability Deficit** | Real-time detection across social-media platforms demands lightweight yet accurate architectures — a balance existing solutions fail to achieve. |
| **Explainability Absence** | Black-box detection models cannot localise or visualise manipulated regions, undermining forensic accountability. |

> 💡 **Core Insight:** A forensically useful deepfake detector must operate across all three modalities simultaneously, deliver real-time inference, and produce interpretable, evidence-grade outputs.

---

## 3. Existing Systems

### 3.1 Survey of Prior Art

| Reference | Approach | Modality | Dataset | Limitation |
|---|---|---|---|---|
| Zhou et al. (2017) | Two-stream CNN for tampered face detection | Image | Custom | Foundational baseline only |
| Li et al. (2018) | Eye-blinking pattern analysis | Video | Custom | Limited to blink artefacts |
| Afchar et al. (2018) — MesoNet | Compact CNN on mesoscopic facial properties | Video | FaceForensics | Fast but lower accuracy |
| Rossler et al. (2019) — FF++ | CNN trained on FaceForensics++ benchmark | Video | FaceForensics++ | Widely used, single-modality |
| Li et al. (2020) — Celeb-DF | High-quality celebrity deepfake dataset + baseline | Video | Celeb-DF | Dataset contribution only |
| Hamza et al. (2022) | MFCC features + ML for audio deepfake detection | Audio | ASVspoof 2019 | Audio-only, ~95% |
| Intel FakeCatcher (2022) | rPPG signals via OpenVINO + computer vision | Video | Proprietary | 96% accuracy, video-only |
| Mittal et al. (2020) | Audio-visual detection via emotion cues | Audio/Video | DFDC | Emotion-level fusion only |

### 3.2 Key Limitations of Existing Approaches

- **Single-modality focus** — no system addresses video, audio, and image manipulation simultaneously.
- **Over-fitting** — models trained on narrow datasets fail to generalise across real-world deepfake variants (e.g., StyleGAN2, FaceSwap).
- **No real-time capability** — computationally heavy architectures are incompatible with live-stream deployment.
- **Fragmented pipelines** — practitioners must maintain separate tools for each modality, increasing operational overhead.
- **No explainability** — outputs lack spatial localisation of manipulated regions, limiting forensic utility.

---

## 4. Proposed System — Sach-AI

### 4.1 Framework Overview

**Sach-AI** is a pioneering multimodal deepfake detection framework (Sar et al., IEEE Access, 2025) that integrates three specialised detection pipelines under a single neural architecture, enabling real-time detection across video, audio, and image content simultaneously.

### 4.2 Per-Modality Performance

| Modality | Core Algorithm(s) | Training Dataset(s) | Accuracy |
|---|---|---|---|
| **Video** | Eulerian Video Magnification + ResNext + LSTM | FaceForensics++, Celeb-DF | **97.76%** |
| **Audio** | Mel Spectrogram + CNN (Conv → MaxPool → BatchNorm → GAP → Dense) | ASVspoof 2019 | **99.13%** |
| **Image** | Fine-tuned DenseNet121 + Binary Cross-Entropy | Flickr 70k + StyleGAN 70k | **93.64%** |

### 4.3 Comparison: Sach-AI vs. Existing Systems

| Parameter | Existing Systems | Sach-AI |
|---|---|---|
| Modality Coverage | Single modality | Video + Audio + Image |
| Video Accuracy | ~96% (Intel FakeCatcher) | **97.76%** (FF++) / **94.66%** (CelebDF) |
| Audio Accuracy | ~95% (MFCC-ML) | **99.13%** (ASVspoof 2019) |
| Explainability | Not available | Grad-CAM heatmap localisation |
| Real-time Capable | Partial / offline only | ✅ Designed for live streams |
| Unified Pipeline | Separate per-modality tools | ✅ Single integrated framework |

### 4.4 Key Novelties

- **First true multimedia framework** — combines video, audio, and image detection under one architecture.
- **EVM + ResNext fusion** — unique algorithm delivering accuracy surpassing Intel FakeCatcher.
- **Mel Spectrogram approach** — improved robustness over conventional MFCC-based audio detection.
- **Fine-tuned DenseNet121** — trained on a custom 140k-image dataset, outperforming existing architectures on advanced GAN-generated fakes.
- **Real-time inference** — designed for deployment on live content streams and web-based moderation platforms.
- **Explainability via Grad-CAM** — spatially localises manipulated facial regions for forensic use.
- **Blockchain roadmap** — future integration for deepfake-creator attribution and origin tracing.

---

## 5. System Architecture

### 5.1 High-Level Architecture

```text
INPUT (Video / Audio / Image)
         │
         ▼
┌─────────────────────────────┐
│  Module 1: Data Ingestion   │
│  & Preprocessing            │
└─────────────┬───────────────┘
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
┌────────┐ ┌──────┐ ┌────────┐
│ Video  │ │Audio │ │ Image  │
│Branch  │ │Branch│ │Branch  │
│(Mod 2) │ │(Mod 3│ │(Mod 4) │
└───┬────┘ └──┬───┘ └───┬────┘
    └─────────┼──────────┘
              ▼
┌──────────────────────────────────┐
│  Module 5: Multimodal Fusion     │
│  & Forensic Report Generation   │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Module 6: UI Dashboard & API   │
└──────────────┬───────────────────┘
               ▼
  OUTPUT: Authenticity Verdict
  + Confidence Score + Heatmaps
```

### 5.2 Three-Branch Processing Pipeline

| Stage | Video Branch | Audio Branch | Image Branch |
|---|---|---|---|
| **Input** | Frame sequence V={I₁…Iₙ} | Audio segments A={a₁…aₙ} | RGB image I |
| **Preprocessing** | EVM spatial decomposition + temporal band-pass filtering | Mel Filterbank → Mel Spectrogram → log compression | Resize, normalise, augment |
| **Feature Extraction** | ResNext (cardinality-enhanced grouped convolutions) | CNN: Conv → MaxPool → BatchNorm → ReLU → Dropout → GAP | DenseNet121 (pre-trained ImageNet, fine-tuned) |
| **Classification** | Stacked LSTM (forget / input / output gates) | Dense layer with sigmoid activation | Binary cross-entropy with softmax output |
| **Output** | Real/Fake + per-frame confidence score | Real/Fake + probability score | Real/Fake + Grad-CAM heatmap |
| **Fusion** | Weighted multimodal fusion layer → unified **Authenticity Verdict** | | |

### 5.3 Key Algorithmic Components

- **EVM Spatial Decomposition** — amplifies subtle per-pixel colour variations to expose GAN artefacts invisible to the naked eye.
- **EVM Temporal Filtering** — measures the rate of colour change over time, highlighting temporal anomalies from deepfake generators.
- **ResNext Feature Extraction** — cardinality-enhanced grouped convolutions with batch normalisation produce rich spatial-channel feature maps.
- **LSTM Sequential Modelling** — forget, input, and output gates capture long-range temporal dependencies across video frame sequences.
- **CNN on Mel Spectrogram** — local spectral pattern extraction from 2-D spectrogram representations of audio signals.
- **DenseNet121 Dense Connectivity** — each layer receives feature maps from all preceding layers, maximising gradient flow and feature reuse.
- **Binary Cross-Entropy Loss** — minimises classification error for real vs. fake binary output across all modalities during training.

---

## 6. Module Descriptions

### ⚙️ Module 1 — Data Ingestion & Preprocessing

Accepts raw video files, audio recordings, and images via web upload or live stream. Performs format validation, face detection, and audio segmentation.

**Video Processing:**
- Frames extracted and resized to **224 × 224 pixels**
- Face bounding boxes detected via **dlib / MTCNN** for region-of-interest cropping

**Audio Processing:**
- Resampled to **16 kHz**
- Segmented into overlapping windows for temporal analysis

**Image Processing:**
- Normalised using **ImageNet mean and standard deviation**
- Face regions cropped for focused analysis

---

### 🎬 Module 2 — Video Deepfake Detection (EVM + ResNext + LSTM)

The video analysis pipeline leverages **Eulerian Video Magnification (EVM)** to amplify temporal colour variations that GAN-generated videos fail to reproduce authentically.

**Pipeline:**
```text
Raw Video Frames
    → EVM Decomposition (colour channels + temporal band-pass filtering)
    → ResNext Feature Extraction (grouped convolutions with cardinality)
    → Stacked LSTM (temporal dependency modelling across frame sequence)
    → Binary Classification: Real / Fake
    → Per-frame confidence score output
```

**Performance:**
- **97.76%** accuracy on FaceForensics++ (100-frame experiments)
- **94.66%** accuracy on Celeb-DF

---

### 🔊 Module 3 — Audio Deepfake Detection (Mel Spectrogram + CNN)

Converts raw audio into 2-D **Mel spectrograms** that encode perceptual frequency characteristics. A CNN stack extracts local spectral patterns, and regularisation layers prevent over-fitting.

**Pipeline:**
```text
Raw Audio
    → Mel Filterbank → 2-D Mel Spectrogram (log compressed)
    → Conv Layers (spectral artefact detection)
    → MaxPooling → BatchNorm → ReLU → Dropout (p=0.5)
    → Global Average Pooling → 1-D descriptor vector
    → Dense Sigmoid Layer → P(deepfake)
    → Threshold 0.5 → Binary Verdict
```

**Performance:** **99.13%** accuracy on ASVspoof 2019

---

### 🖼️ Module 4 — Image Deepfake Detection (DenseNet121)

A fine-tuned **DenseNet121** model, originally pre-trained on ImageNet, is further trained on 142,041 real and fake face images. Dense connectivity ensures maximum feature reuse and gradient flow.

**Pipeline:**
```text
RGB Image
    → Resize + Normalise (ImageNet stats)
    → DenseNet121 (dense blocks: growth rate k, each layer → all subsequent layers)
    → Fine-tuned on 70k real (Flickr) + 70k fake (StyleGAN) images
    → Binary Cross-Entropy Loss (Adam optimiser)
    → Binary Classification: Real / Fake
    → Grad-CAM Heatmap (localises manipulated facial regions)
```

**Performance:**
- **93.64%** accuracy
- ROC-AUC: **0.992**
- Average Precision: **0.991**

---

### 📊 Module 5 — Unified Multimodal Fusion & Reporting

Aggregates predictions from all three branches via a **weighted fusion layer** and generates a comprehensive forensic authenticity report.

**Fusion Weights (tunable per use case):**

| Branch | Default Weight |
|---|---|
| Video | 0.4 |
| Audio | 0.3 |
| Image | 0.3 |

**Report Contents:**
- Final verdict: **AUTHENTIC** or **DEEPFAKE** with overall confidence percentage
- Per-modality verdicts and individual confidence scores
- Timestamped metadata and flagged segments
- Visual evidence: Grad-CAM heatmaps (image), Mel spectrograms (audio), confidence plots (video)
- JSON API response for integration with content moderation platforms

---

### 🚀 Module 6 — User Interface & API Gateway

Provides a web-based dashboard and REST API for end-users and platform integrators, supporting both batch analysis and live-stream real-time detection.

**Deployment Modes:**

**A. Web Dashboard**
- Drag-and-drop file upload
- Live webcam / microphone stream via WebSocket
- Real-time inference display with visual report and heatmaps

**B. REST API**

*Request:*
```json
{
  "media": "base64_encoded_file",
  "type": "video | audio | image"
}
```

*Response:*
```json
{
  "verdict": "DEEPFAKE",
  "confidence": 0.94,
  "video_score": 0.96,
  "audio_score": 0.98,
  "image_score": 0.88,
  "heatmap_url": "https://example.com/heatmap/xyz.png",
  "timestamp": "2026-05-15T10:30:00Z"
}
```

**Features:** API key authentication · Rate limiting · Pre-loaded model weights for fast inference

---

## 7. End-to-End Workflow

### 7.1 Detailed Processing Flow

```text
Step 1 │ UPLOAD
        User submits a suspicious video via the web interface

Step 2 │ MODULE 1 — INGESTION
        Extract frames + audio → Detect faces → Preprocess all modalities

Step 3 │ MODULE 2 — VIDEO ANALYSIS
        EVM amplifies GAN artefacts
        → ResNext extracts spatial features
        → LSTM classifies frame sequence
        → Result: 97% fake confidence

Step 4 │ MODULE 3 — AUDIO ANALYSIS
        Convert audio to Mel spectrogram
        → CNN detects synthetic voice patterns
        → Result: 99% fake confidence

Step 5 │ MODULE 4 — IMAGE ANALYSIS
        DenseNet121 analyses key frames
        → Grad-CAM highlights manipulated facial regions
        → Result: 93% fake confidence

Step 6 │ MODULE 5 — FUSION
        Weighted combination:
        (0.4 × 0.97) + (0.3 × 0.99) + (0.3 × 0.93) = 0.96 overall confidence

Step 7 │ MODULE 6 — OUTPUT
        Verdict: DEEPFAKE (96% confidence)
        → Display Grad-CAM heatmap
        → Show Mel spectrogram anomalies
        → Highlight suspicious video frames

Step 8 │ INVESTIGATION
        Forensic investigator reviews structured evidence report
```

### 7.2 Technology Pipeline Summary

| Stage | Video | Audio | Image |
|---|---|---|---|
| **Input** | Frame sequence | Audio segments | RGB image |
| **Preprocessing** | EVM magnification | Mel spectrogram | Resize + normalise |
| **Feature Extraction** | ResNext CNN | CNN layers | DenseNet121 |
| **Classification** | LSTM | Dense sigmoid | Binary cross-entropy |
| **Output** | Confidence/frame | Probability score | Grad-CAM heatmap |
| **Final Step** | **Weighted fusion → Unified authenticity verdict** | | |

---

## 8. System Requirements

### 8.1 Hardware Requirements

| Component | Specification |
|---|---|
| **Processor** | Intel Core i7 / AMD Ryzen 7 or above |
| **GPU** | NVIDIA GTX 1660 Ti / RTX 2060 (minimum 6 GB VRAM) |
| **RAM** | 16 GB DDR4 minimum (32 GB recommended for training) |
| **Storage** | SSD with at least 200 GB free (datasets + model checkpoints) |
| **Input Devices** | HD webcam and microphone for real-time stream acquisition |
| **Network** | Broadband internet for dataset downloads and cloud deployment |

### 8.2 Software Requirements

| Category | Tools / Libraries |
|---|---|
| **OS** | Ubuntu 20.04 LTS / Windows 10 (64-bit) |
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow 2.x / PyTorch 1.13+ |
| **Computer Vision** | OpenCV 4.x |
| **Data Handling** | NumPy, Pandas, scikit-learn |
| **Audio Processing** | Librosa, SciPy |
| **Visualisation** | Matplotlib, Seaborn, Grad-CAM library |
| **Model Serving** | Flask / FastAPI |
| **IDE** | VS Code / Jupyter Notebook / Google Colab |
| **Version Control** | Git + GitHub |

### 8.3 Dataset Requirements

| Dataset | Description |
|---|---|
| **FaceForensics++ (Video)** | 1,000 original videos with 4 manipulation methods (Deepfakes, FaceSwap, Face2Face, NeuralTextures); sourced from YouTube |
| **Celeb-DF (Video)** | 5,630 high-quality celebrity deepfake videos generated via an improved synthesis pipeline |
| **Flickr + StyleGAN (Image)** | 70k real faces (Flickr/NVIDIA) + 70k fake faces (StyleGAN); total 140k images for training and validation |
| **ASVspoof 2019 (Audio)** | Logical and physical access tracks from VCTK corpus; includes TTS, voice conversion, and replay attacks |

---

## 9. Technology Stack

| Component | Technology / Library |
|---|---|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch / TensorFlow + Keras |
| **Face Detection** | MTCNN / OpenCV Haar Cascade / dlib |
| **CNN Architecture** | Custom CNN + EfficientNet-B4 / DenseNet121 / ResNext (Transfer Learning) |
| **Temporal Modelling** | Stacked LSTM |
| **Frequency Analysis** | NumPy FFT, OpenCV DCT, Eulerian Video Magnification |
| **Audio Processing** | Librosa (Mel Spectrogram), SciPy |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualisation** | Matplotlib, Seaborn, Grad-CAM heatmaps |
| **Datasets** | FaceForensics++, DFDC, CelebDF, ASVspoof 2019, Flickr + StyleGAN |
| **Web Interface** | Streamlit / Flask / FastAPI |
| **IDE** | Jupyter Notebook / VS Code / Google Colab |

---

## 10. Conclusion

**Sach-AI** represents a significant step forward in the fight against synthetic media manipulation. By unifying three independent detection pipelines — video, audio, and image — under a single neural architecture, the system addresses the core limitations of all prior single-modality approaches.

Key achievements of the proposed framework:

- **Superior accuracy** across all modalities, surpassing Intel FakeCatcher's 96% benchmark for video deepfakes.
- **Real-time inference capability**, enabling deployment on live content streams.
- **Explainable outputs** through Grad-CAM heatmaps, providing forensically admissible evidence.
- **Unified pipeline**, eliminating the need for separate per-modality tools in operational environments.

> The system is designed not merely as a research prototype, but as a **practical, deployable forensic tool** — capable of assisting digital investigators, content moderators, and law enforcement agencies in verifying media authenticity in an increasingly deepfake-saturated information landscape.

Future work includes blockchain-based attribution to trace deepfake origin and creator identity, further strengthening the system's utility within the digital forensics domain.

---

## 11. Additional Notes & Data

- **Loopback:** 127.0.0.0
- **Features:** Audio signatures, TicToc
- **Metrics:** Starting time
- **Outputs:** Data from model, Gist

---

*Documentation prepared for CIP Review | SCSVMV University — Department of CSE | Jan 2026 – May 2026*
