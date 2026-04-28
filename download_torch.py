"""
download_torch.py — Resumable downloader for PyTorch wheel + install
Run: python download_torch.py

Downloads torch, torchvision, torchaudio wheels to a local cache folder
and installs them — bypasses pip timeout on large files.
"""

import urllib.request
import os
import sys
import subprocess
from pathlib import Path

# ── Wheels to download ───────────────────────────────────────────────────────
BASE = "https://download.pytorch.org/whl/cu121"

WHEELS = [
    ("torch-2.5.1+cu121-cp311-cp311-win_amd64.whl",
     f"{BASE}/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"),
    ("torchvision-0.20.1+cu121-cp311-cp311-win_amd64.whl",
     f"{BASE}/torchvision-0.20.1%2Bcu121-cp311-cp311-win_amd64.whl"),
    ("torchaudio-2.5.1+cu121-cp311-cp311-win_amd64.whl",
     f"{BASE}/torchaudio-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"),
]

CACHE = Path(__file__).parent.parent / "torch_wheels"
CACHE.mkdir(exist_ok=True)

print(f"\nWheel cache directory: {CACHE.resolve()}\n")


def show_progress(count, block_size, total):
    done = count * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        mb_done = done / 1_048_576
        mb_total = total / 1_048_576
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {mb_done:.0f}/{mb_total:.0f} MB")
        sys.stdout.flush()


def download_resumable(url: str, dest: Path):
    """Download with byte-range resume support."""
    headers = {}
    existing = dest.stat().st_size if dest.exists() else 0

    # HEAD request to get total size
    req_head = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req_head, timeout=30) as r:
            total = int(r.headers.get("Content-Length", 0))
    except Exception:
        total = 0

    if existing > 0 and total > 0 and existing >= total:
        print(f"  Already complete ({existing / 1_048_576:.0f} MB) — skipping download.")
        return
    elif existing > 0 and total > 0:
        print(f"  Resuming from {existing / 1_048_576:.0f} MB / {total / 1_048_576:.0f} MB")
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"

    CHUNK = 1024 * 1024  # 1 MB chunks
    downloaded = existing

    try:
        with urllib.request.urlopen(req, timeout=120) as response, \
             open(dest, mode) as out:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = min(100, downloaded * 100 // total)
                    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                    mb = downloaded / 1_048_576
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:3d}%  {mb:.0f}/{total/1_048_576:.0f} MB"
                    )
                    sys.stdout.flush()
    except Exception as e:
        print(f"\n  [WARN] Download interrupted: {e}. Re-run to resume.")
        sys.exit(1)

    print(f"\n  Done -> {dest.name}")


def pip_install(wheel_path: Path):
    cmd = [sys.executable, "-m", "pip", "install", str(wheel_path),
           "--no-deps", "--force-reinstall"]
    print(f"\n  Installing {wheel_path.name}...")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    print("=" * 60)
    print("  PyTorch GPU Wheel Downloader (Resumable)")
    print("=" * 60)

    wheel_paths = []
    for filename, url in WHEELS:
        dest = CACHE / filename
        print(f"\nWheel: {filename}")
        if not dest.exists() or dest.stat().st_size < 1_000_000:
            print(f"  URL : {url}")
            download_resumable(url, dest)
        else:
            # Check if partial
            existing = dest.stat().st_size
            print(f"  Found: {existing / 1_048_576:.0f} MB on disk")
            req_head = urllib.request.Request(url, method="HEAD")
            try:
                with urllib.request.urlopen(req_head, timeout=30) as r:
                    total = int(r.headers.get("Content-Length", 0))
                if existing < total:
                    print(f"  Partial ({existing/1_048_576:.0f}/{total/1_048_576:.0f} MB) — resuming...")
                    download_resumable(url, dest)
                else:
                    print(f"  Complete — skipping download.")
            except Exception:
                download_resumable(url, dest)
        wheel_paths.append(dest)

    print("\n" + "=" * 60)
    print("  All wheels downloaded. Installing...")
    print("=" * 60)

    for wp in wheel_paths:
        pip_install(wp)

    print("\n" + "=" * 60)
    print("  PyTorch installation complete!")
    print("\n  Verifying...")
    subprocess.check_call([sys.executable, "-c",
        "import torch; print(f'  torch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}')"
    ])
    print("=" * 60 + "\n")
