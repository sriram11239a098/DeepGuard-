"""
utils/inference.py — Inference engine for DeepGuard / Sach-AI.

Loads the unified MultimodalFusionModel and provides a single
`.predict(file_path)` method that returns a rich FusionResult.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

import cv2
import torch

import config
from modules.fusion_model import MultimodalFusionModel
from modules.preprocessing import extract_frames, extract_audio_mel
from modules.transforms import get_image_transforms, get_video_transforms
from utils.tictoc import TicToc
from utils.evm import apply_evm, get_evm_diff_map
from utils.gradcam import GradCAM, overlay_heatmap

# ── Supported extensions ──────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Result data classes ───────────────────────────────────────────────────────

@dataclass
class ModalityScore:
    score: Optional[float] = None


@dataclass
class FusionResult:
    verdict:        str
    confidence_pct: float
    video:          ModalityScore = field(default_factory=ModalityScore)
    audio:          ModalityScore = field(default_factory=ModalityScore)
    image:          ModalityScore = field(default_factory=ModalityScore)
    start_time:     str           = "N/A"
    end_time:       str           = "N/A"
    audio_sig:      str           = "N/A"
    model_data:     dict          = field(default_factory=dict)
    gist:           str           = ""
    heatmap_path:   Optional[str] = None
    elapsed_seconds: float         = 0.0

    def to_dict(self) -> dict:
        return {
            "verdict":        self.verdict,
            "confidence_pct": round(self.confidence_pct, 2),
            "video_score":    self.video.score,
            "audio_score":    self.audio.score,
            "image_score":    self.image.score,
            "metrics": {
                "start_time": self.start_time,
                "end_time":   self.end_time,
            },
            "audio_signature": self.audio_sig,
            "model_data":      self.model_data,
            "gist":            self.gist,
            "heatmap_path":    self.heatmap_path,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Inference engine ──────────────────────────────────────────────────────────

class DeepGuardInference:
    """
    Wraps MultimodalFusionModel for single-file deepfake detection.

    Usage:
        engine = DeepGuardInference(load_pretrained_weights=True)
        result = engine.predict("path/to/file.mp4")
    """

    def __init__(self, load_pretrained_weights: bool = True):
        self.device = config.DEVICE
        print(f"\n[DeepGuard] Initialising unified model on: {self.device}")

        self.model = MultimodalFusionModel(
            hidden_dim=config.FUSION_HIDDEN_DIM
        ).to(self.device)

        ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_fusion_model.pth")
        if load_pretrained_weights and os.path.exists(ckpt):
            checkpoint = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[DeepGuard] Loaded checkpoint: {ckpt}")
        else:
            print("[DeepGuard] WARNING — no trained checkpoint found; using random weights.")

        self.model.eval()

        self._img_tf  = get_image_transforms(config.IMAGE_SIZE, is_train=False)
        self._vid_tf  = get_video_transforms(config.IMAGE_SIZE, is_train=False)
        
        # Initialize GradCAM for Image branch
        try:
            self.gradcam = GradCAM(self.model.image_model, target_layer_name="backbone.features")
        except Exception:
            self.gradcam = None
            
        print("[DeepGuard] Model ready.\n")

    # ── Expected audio tensor width ───────────────────────────────────────────
    @property
    def _audio_steps(self) -> int:
        return int(config.AUDIO_SR * config.AUDIO_DURATION / 512) + 1

    # ── Main predict method ───────────────────────────────────────────────────

    def predict(self, file_path: str) -> FusionResult:
        """
        Run deepfake detection on any supported media file.

        The modality is inferred from the file extension.
        Returns a FusionResult populated with confidence scores and metadata.
        """
        timer = TicToc()
        timer.tic()

        p   = Path(file_path)
        ext = p.suffix.lower()

        if   ext in VIDEO_EXTS: modality = "video"
        elif ext in AUDIO_EXTS: modality = "audio"
        elif ext in IMAGE_EXTS: modality = "image"
        else:
            timer.toc()
            return FusionResult(
                verdict="UNKNOWN", confidence_pct=0.0,
                gist=f"Unsupported extension: {ext}"
            )

        # ── Build input tensors (all zero except the active modality) ─────────
        img_t   = torch.zeros(
            1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=self.device
        )
        vid_t   = torch.zeros(
            1, config.VIDEO_MAX_FRAMES, 3,
            config.IMAGE_SIZE, config.IMAGE_SIZE, device=self.device
        )
        aud_t   = torch.zeros(
            1, 1, config.N_MELS, self._audio_steps, device=self.device
        )
        audio_sig = "N/A"

        # ── Preprocessing (No Gradients needed) ───────────────────────────────
        with torch.no_grad():
            if modality == "video":
                print(f"[Inference] Video  → {p.name}")
                frames = extract_frames(str(p), config.VIDEO_MAX_FRAMES)
                t_frames = [self._vid_tf(image=f)["image"] for f in frames]
                if t_frames:
                    vid_t[0] = torch.stack(t_frames).to(self.device)
                    img_t[0] = self._img_tf(image=frames[len(frames) // 2])["image"].to(self.device)
                try:
                    mel = extract_audio_mel(
                        str(p), config.AUDIO_SR, config.AUDIO_DURATION, config.N_MELS
                    )
                    aud_t[0] = mel.to(self.device)
                    audio_sig = hashlib.sha256(
                        mel.cpu().numpy().tobytes()
                    ).hexdigest()[:16]
                except Exception:
                    pass

            elif modality == "image":
                print(f"[Inference] Image  → {p.name}")
                img = cv2.imread(str(p))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_t[0] = self._img_tf(image=img)["image"].to(self.device)

            elif modality == "audio":
                print(f"[Inference] Audio  → {p.name}")
                try:
                    mel = extract_audio_mel(
                        str(p), config.AUDIO_SR, config.AUDIO_DURATION, config.N_MELS
                    )
                    aud_t[0] = mel.to(self.device)
                    audio_sig = hashlib.sha256(
                        mel.cpu().numpy().tobytes()
                    ).hexdigest()[:16]
                except Exception:
                    pass

        # ── Heatmap & EVM Generation ──────────────────────────────────────
        heatmap_path = None
        
        if modality == "image" and self.gradcam:
            try:
                # Grad-CAM REQUIRES gradients
                img_t.requires_grad = True
                with torch.set_grad_enabled(True):
                    cam = self.gradcam.generate(img_t)
                
                orig_img = cv2.imread(str(p))
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                viz = overlay_heatmap(orig_img, cam)
                
                h_name = f"heatmap_{p.stem}.png"
                h_path = config.OUTPUTS_DIR / h_name
                cv2.imwrite(str(h_path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                heatmap_path = str(h_path)
                img_t.requires_grad = False # Clean up
            except Exception as e:
                print(f"[Inference] Grad-CAM failed: {e}")

        elif modality == "video":
            try:
                with torch.no_grad():
                    frames = extract_frames(str(p), config.VIDEO_MAX_FRAMES)
                    viz = get_evm_diff_map(frames)
                    
                    h_name = f"evm_map_{p.stem}.png"
                    h_path = config.OUTPUTS_DIR / h_name
                    cv2.imwrite(str(h_path), viz)
                    heatmap_path = str(h_path)
            except Exception as e:
                print(f"[Inference] EVM map failed: {e}")

        # ── Forward pass — Final Authenticity Verdict (No Gradients) ──────────
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(img_t, aud_t, vid_t)
            else:
                logits = self.model(img_t, aud_t, vid_t)

            prob = torch.sigmoid(logits).item()

        elapsed = timer.toc()
        start, end = timer.get_times()

        verdict        = "DEEPFAKE" if prob >= config.DETECTION_THRESHOLD else "AUTHENTIC"
        confidence_pct = prob * 100

        # Populate per-modality scores sensibly
        vid_score = round(prob, 4) if modality == "video"                     else None
        aud_score = round(prob, 4) if modality in ("audio", "video")          else None
        img_score = round(prob, 4) if modality in ("image", "video")          else None

        return FusionResult(
            verdict        = verdict,
            confidence_pct = round(confidence_pct, 2),
            video          = ModalityScore(vid_score),
            audio          = ModalityScore(aud_score),
            image          = ModalityScore(img_score),
            start_time     = start,
            end_time       = end,
            audio_sig      = audio_sig,
            model_data     = {
                "raw_probability": round(prob, 6),
                "modality":        modality,
                "device":          self.device,
            },
            gist = (
                f"'{p.name}' analysed as {verdict} "
                f"({confidence_pct:.1f}% confidence) via {modality} branch."
            ),
            heatmap_path = heatmap_path,
            elapsed_seconds = elapsed,
        )
