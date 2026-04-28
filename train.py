"""
train.py — Training script for DeepGuard / Sach-AI.

Run via:
    python train.py
    python main.py train --modality all
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.dataset import MultimodalDeepfakeDataset
from modules.fusion_model import MultimodalFusionModel
from utils.logger import setup_logger
from utils.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# One epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler,
    accumulation_steps, device, use_amp
):
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="  Train", leave=False)

    for i, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        videos = batch["video"].to(device, non_blocking=True)
        audios = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images, audios, videos)
                loss   = criterion(logits, labels) / accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits = model(images, audios, videos)
            loss   = criterion(logits, labels) / accumulation_steps
            loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().ravel().tolist())

        pbar.set_postfix({"loss": f"{running_loss / (i + 1):.4f}"})

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return running_loss / max(len(dataloader), 1), metrics


def validate(model, dataloader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Val  ", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            videos = batch["video"].to(device, non_blocking=True)
            audios = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(images, audios, videos)
                    loss   = criterion(logits, labels)
            else:
                logits = model(images, audios, videos)
                loss   = criterion(logits, labels)

            running_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().ravel().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return running_loss / max(len(dataloader), 1), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger = setup_logger("training", config.LOGS_DIR)
    device = torch.device(config.DEVICE)
    logger.info(f"Using device: {device}")

    if not config.DATASET_ROOT_DIR.exists():
        logger.error(
            f"Dataset root not found: {config.DATASET_ROOT_DIR}. "
            "Please update DATASET_ROOT_DIR in config.py."
        )
        return

    # ── Check for ffmpeg ──────────────────────────────────────────────────────
    import shutil
    if not shutil.which("ffmpeg"):
        logger.warning(
            "ffmpeg.exe not found in system PATH. Audio cannot be extracted "
            "from video files. Video branches will train on silence. "
            "Install ffmpeg (https://ffmpeg.org) for full multimodal support."
        )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = MultimodalDeepfakeDataset(config.DATASET_ROOT_DIR, split="train")
    val_dataset   = MultimodalDeepfakeDataset(config.DATASET_ROOT_DIR, split="val")

    logger.info(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty — check your dataset paths and folder layout.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = (config.DEVICE == "cuda"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = (config.DEVICE == "cuda"),
    )

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = MultimodalFusionModel(hidden_dim=config.FUSION_HIDDEN_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.LEARNING_RATE),
        weight_decay=float(config.WEIGHT_DECAY),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    # Mixed precision only on CUDA
    use_amp = config.MIXED_PRECISION and (config.DEVICE == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    latest_ckpt = os.path.join(config.CHECKPOINTS_DIR, "latest_fusion_model.pth")
    best_ckpt   = os.path.join(config.CHECKPOINTS_DIR, "best_fusion_model.pth")

    # ── Resume from checkpoint if available ───────────────────────────────────
    start_epoch  = 0
    best_val_auc = 0.0
    resume_path  = latest_ckpt if os.path.exists(latest_ckpt) else (
                   best_ckpt   if os.path.exists(best_ckpt)   else None)

    if resume_path:
        logger.info(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_auc = ckpt.get("val_auc", 0.0)
        logger.info(f"Resumed at epoch {start_epoch}, best_val_auc={best_val_auc:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    patience_counter = 0
    accum_steps      = config.GRADIENT_ACCUMULATION_STEPS

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        logger.info(f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}]")

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            accum_steps, device, use_amp
        )
        val_loss, val_m = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step(val_m["auc"])

        logger.info(
            f"  Train → loss={train_loss:.4f}  AUC={train_m['auc']:.4f}  "
            f"Acc={train_m['accuracy']:.4f}  F1={train_m['f1']:.4f}"
        )
        logger.info(
            f"  Val   → loss={val_loss:.4f}  AUC={val_m['auc']:.4f}  "
            f"Acc={val_m['accuracy']:.4f}  F1={val_m['f1']:.4f}"
        )

        # Always save the latest checkpoint for safe resumption
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc":              best_val_auc,
        }, latest_ckpt)

        # Save best checkpoint
        if val_m["auc"] > best_val_auc:
            best_val_auc     = val_m["auc"]
            patience_counter = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc":              best_val_auc,
            }, best_ckpt)
            logger.info(f"  ✓ New best model saved (AUC={best_val_auc:.4f}) → {best_ckpt}")
        else:
            patience_counter += 1
            logger.info(
                f"  No improvement. Patience {patience_counter}/{config.PATIENCE}"
            )

        if patience_counter >= config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
