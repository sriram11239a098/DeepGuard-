"""
main.py — DeepGuard / Sach-AI  |  Single CLI entrypoint

Modes:
    python main.py train   [--modality video|audio|image|all]
    python main.py predict --file <path_to_media>
    python main.py ui                ← launch Streamlit dashboard
    python main.py api               ← launch FastAPI server
    python main.py check             ← verify environment & dataset paths
"""

import argparse
import io
import subprocess
import sys

# ── Force UTF-8 output on Windows ────────────────────────────────────────────
try:
    if (sys.stdout.encoding or "").lower() != "utf-8" and not getattr(sys.stdout, "closed", False):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if (sys.stderr.encoding or "").lower() != "utf-8" and not getattr(sys.stderr, "closed", False):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import config


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: check
# ─────────────────────────────────────────────────────────────────────────────

def cmd_check():
    import torch
    print("\n── Environment Check ─────────────────────────────────────")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Device used : {config.DEVICE}")
    import shutil
    ff = shutil.which("ffmpeg")
    print(f"  ffmpeg      : {'✓ found' if ff else '✗ NOT FOUND'} ({ff or 'N/A'})")


    print(f"\n── Dataset Paths ─────────────────────────────────────────")
    paths = {
        "Video": config.DATASET_VIDEO_DIR,
        "Audio": config.DATASET_AUDIO_DIR,
        "Image": config.DATASET_IMAGE_DIR,
    }
    for name, p in paths.items():
        exists = p.exists()
        real_n = len(list((p / "real").glob("*"))) if (p / "real").exists() else 0
        fake_n = len(list((p / "fake").glob("*"))) if (p / "fake").exists() else 0
        status = "✓" if exists else "✗ NOT FOUND"
        print(f"  {name:<6} {status}  →  {p}")
        if exists:
            print(f"          real: {real_n} files   fake: {fake_n} files")

    print("\n── Checkpoints ───────────────────────────────────────────")
    for name in ["latest_fusion_model", "best_fusion_model"]:
        p = config.CHECKPOINTS_DIR / f"{name}.pth"
        print(f"  {name}: {'✓ found' if p.exists() else '✗ not trained yet'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: predict
# ─────────────────────────────────────────────────────────────────────────────

def cmd_predict(file_path: str):
    from utils.inference import DeepGuardInference
    engine = DeepGuardInference(load_pretrained_weights=True)
    result = engine.predict(file_path)
    print("\n" + "=" * 50)
    print(f"  VERDICT     : {result.verdict}")
    print(f"  CONFIDENCE  : {result.confidence_pct:.1f}%")
    print(f"  Video score : {result.video.score}")
    print(f"  Audio score : {result.audio.score}")
    print(f"  Image score : {result.image.score}")
    print(f"  Start time  : {result.start_time}")
    print(f"  End time    : {result.end_time}")
    print("=" * 50)
    print("\nFull JSON report:")
    print(result.to_json())


# ─────────────────────────────────────────────────────────────────────────────
# Subcommands: ui / api
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ui():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port",    str(config.UI_PORT),
        "--server.address", config.API_HOST,
    ], cwd=config.PROJECT_ROOT)


def cmd_api():
    subprocess.run([
        sys.executable, "-m", "uvicorn", "api:app",
        "--host",   config.API_HOST,
        "--port",   str(config.API_PORT),
        "--reload",
    ], cwd=config.PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepGuard / Sach-AI — Multimodal Deepfake Detection"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train the fusion model")
    p_train.add_argument(
        "--modality",
        choices=["video", "audio", "image", "all"],
        default="all",
        help="(informational — all modalities are always trained together)",
    )

    p_pred = sub.add_parser("predict", help="Run inference on a media file")
    p_pred.add_argument("--file", required=True, help="Path to media file")

    sub.add_parser("ui",    help="Launch Streamlit dashboard")
    sub.add_parser("api",   help="Launch FastAPI REST server")
    sub.add_parser("check", help="Verify environment and dataset paths")

    args = parser.parse_args()

    if args.mode == "train":
        subprocess.run([sys.executable, "train.py"], cwd=config.PROJECT_ROOT)

    elif args.mode == "predict":
        cmd_predict(args.file)

    elif args.mode == "ui":
        cmd_ui()

    elif args.mode == "api":
        cmd_api()

    elif args.mode == "check":
        cmd_check()
