"""
arrange_datasets.py — One-time utility to organise raw datasets into the
DeepGuard real/fake directory layout.

Edit SOURCE_DIR and TARGET_DIR to match your environment, then run once:
    python arrange_datasets.py
"""

import os
import random
import shutil
from pathlib import Path

# ── Configure these paths ─────────────────────────────────────────────────────
SOURCE_DIR = Path(r"E:\CIP\Datasets")
TARGET_DIR = Path(r"E:\CIP\Datasets_Ready")

MAX_PER_CATEGORY = 7000  # cap per (modality, label) — keeps demo training fast

MODALITIES = {
    "Video": {".mp4", ".avi", ".mov", ".mkv"},
    "Audio": {".wav", ".flac", ".mp3", ".m4a"},
    "Image": {".jpg", ".jpeg", ".png", ".bmp"},
}


def get_modality(suffix: str) -> str | None:
    s = suffix.lower()
    for mod, exts in MODALITIES.items():
        if s in exts:
            return mod
    return None


def get_label(filename: str) -> str | None:
    low = filename.lower()
    if any(k in low for k in ("real", "authentic", "original")):
        return "real"
    if any(k in low for k in ("fake", "spoof", "manipulated", "deepfake")):
        return "fake"
    return None


print(f"Scanning {SOURCE_DIR} …")

collected: dict[str, dict[str, list]] = {
    mod: {"real": [], "fake": [], "unlabelled": []}
    for mod in MODALITIES
}

# Pass 1 — collect
for root, _, files in os.walk(SOURCE_DIR):
    for fname in files:
        mod = get_modality(Path(fname).suffix)
        if not mod:
            continue
        label = get_label(fname)
        full  = os.path.join(root, fname)
        if label:
            collected[mod][label].append(full)
        else:
            collected[mod]["unlabelled"].append(full)

# Pass 2 — fallback split for modalities that have no explicit labels
for mod in MODALITIES:
    if not collected[mod]["real"] and not collected[mod]["fake"] and collected[mod]["unlabelled"]:
        random.seed(42)
        unlabelled = collected[mod]["unlabelled"][:]
        random.shuffle(unlabelled)
        mid = len(unlabelled) // 2
        collected[mod]["real"] = unlabelled[:mid]
        collected[mod]["fake"] = unlabelled[mid:]

# Pass 3 — copy
for mod in MODALITIES:
    for label in ("real", "fake"):
        dest = TARGET_DIR / mod / label
        dest.mkdir(parents=True, exist_ok=True)

        pool = collected[mod][label]
        random.seed(42)
        random.shuffle(pool)
        selected = pool[:MAX_PER_CATEGORY]

        copied = 0
        for src in selected:
            dst = dest / Path(src).name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception as exc:
                    print(f"  [WARN] Could not copy {src}: {exc}")
            else:
                copied += 1

        print(f"[{mod:5s} / {label}]  found={len(pool):4d}  copied={copied:4d}  → {dest}")

print("\nDataset arrangement complete.")
