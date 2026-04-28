# DeepGuard: Multimodal Forensic Deepfake Detection System

DeepGuard (Sach-AI) is a state-of-the-art forensic analysis tool designed to detect deepfakes across multiple modalities including **Video, Audio, and Image**. By utilizing a unified multimodal fusion architecture, the system provides high-confidence verdicts based on the synchronized analysis of facial features, voice signatures, and temporal consistency.

## 🚀 Key Features
- **Multimodal Fusion:** Integrates data from three distinct branches (Image, Audio, Video) for a comprehensive security verdict.
- **Forensic Diagnostics:** Provides detailed scores for each modality and calculates overall confidence using a weighted fusion model.
- **Real-Time Web Dashboard:** A user-friendly Streamlit interface for file uploads, analysis, and result visualization.
- **Robust Training Pipeline:** Features resumable training, mixed-precision (FP16) support for NVIDIA GPUs, and automatic checkpointing.
- **REST API:** Built with FastAPI for seamless integration into existing security infrastructures.

## 🏗️ Technical Architecture
The system employs a "Late Fusion" strategy where individual feature vectors from each modality are concatenated and passed through a final classification layer.

| Branch | Architecture | Purpose |
| :--- | :--- | :--- |
| **Image** | EfficientNet-B0 (DenseNet121) | Static facial artifact detection. |
| **Audio** | ResNet-18 (Custom CNN) | Mel-spectrogram voice signature analysis. |
| **Video** | ResNext-50 + BiGRU | Temporal consistency and frame-to-frame forensic analysis. |
| **Fusion** | Multi-layer Perceptron (MLP) | Learns optimal weights for combining modality scores. |

## 📊 Training Results
The model was trained on a balanced dataset of real and manipulated media.
- **Best Validation AUC:** `0.7243` (Epoch 7)
- **Validation Accuracy:** `70.4%`
- **Device Used:** NVIDIA GeForce RTX 3050 (6GB VRAM)
- **Batch Size:** 12 (Optimized)

## 🛠️ Setup & Installation
1. **Clone the project:**
   ```bash
   git clone https://github.com/yourusername/DeepGuard.git
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python main.py ui
   ```

## 📖 Usage
- **CLI:** `python main.py predict --file "path/to/media.mp4"`
- **UI:** Navigate to `http://localhost:8501` to upload files via the web interface.
- **API:** Use the FastAPI endpoint at `http://localhost:8000/docs` for batch processing.

## 🎓 University Report Context
This project was developed as part of a forensic AI research initiative focused on identifying generative adversarial network (GAN) artifacts and neural speech synthesis anomalies. The multimodal approach significantly reduces the False Acceptance Rate (FAR) compared to unimodal detection systems.

---
**Author:** V Sriram Charan & VKS Karthik   
**System Requirements:** Python 3.11+, NVIDIA CUDA 12.1+ (Recommended)
