"""
install.py — One-shot dependency installer for DeepGuard / Sach-AI
Run:  python install.py
      python install.py --cpu   ← force CPU-only PyTorch
"""
import subprocess
import sys
import argparse

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

def check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="Install CPU-only PyTorch")
args = parser.parse_args()

print("\n" + "="*60)
print("  DeepGuard / Sach-AI  — Dependency Installer")
print("="*60)
print(f"  Python: {sys.version.split()[0]}")

# ── Step 1: Upgrade pip ───────────────────────────────────────
print("\n[1/3] Upgrading pip...")
pip("--upgrade", "pip")

# ── Step 2: PyTorch ───────────────────────────────────────────
print("\n[2/3] Installing PyTorch...")
if args.cpu:
    print("  -> CPU-only mode")
    pip("torch", "torchvision", "torchaudio")
else:
    print("  -> Trying CUDA 12.1 build...")
    try:
        pip("torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121")
        if check_cuda():
            print("  -> CUDA is available!")
        else:
            print("  ⚠ PyTorch installed but CUDA not detected (driver issue or CPU-only GPU)")
    except subprocess.CalledProcessError:
        print("  CUDA 12.1 failed -> falling back to CPU PyTorch")
        pip("torch", "torchvision", "torchaudio")

print("\n[3/4] Installing facenet-pytorch directly...")
pip("facenet-pytorch>=2.6.0", "--no-deps")

# ── Step 4: All other deps ────────────────────────────────────
print("\n[4/4] Installing remaining dependencies...")
pip(
    "opencv-python>=4.9.0.80",
    "librosa>=0.10.2",
    "soundfile>=0.12.1",
    "numpy>=1.26.4",
    "scipy>=1.13.0",
    "pandas>=2.2.2",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.2",
    "Pillow",
    "streamlit>=1.35.0",
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
    "tqdm>=4.66.4",
    "requests>=2.32.3",
)

print("\n" + "="*60)
print("  Installation complete!")
print("  Run:  python main.py check")
print("  UI:   python main.py ui")
print("="*60 + "\n")
