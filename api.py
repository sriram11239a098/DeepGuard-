"""
api.py — FastAPI REST API for DeepGuard / Sach-AI.

Endpoint : POST /detect
Run with : uvicorn api:app --host 0.0.0.0 --port 8000 --reload
        or: python main.py api
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import config
from utils.inference import DeepGuardInference

app = FastAPI(
    title       = "DeepGuard / Sach-AI API",
    description = "Multimodal Deepfake Detection — Video, Audio, Image",
    version     = "1.0.0",
)

# ── Lazy-load the inference engine once on first request ─────────────────────
_engine: DeepGuardInference | None = None


def get_engine() -> DeepGuardInference:
    global _engine
    if _engine is None:
        _engine = DeepGuardInference(load_pretrained_weights=True)
    return _engine


SUPPORTED_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",   # video
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",   # audio
    ".jpg", ".jpeg", ".png", ".bmp",            # image
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "DeepGuard / Sach-AI API is running",
        "device":  config.DEVICE,
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "device": config.DEVICE}


@app.post("/detect", tags=["Inference"])
async def detect(file: UploadFile = File(...)):
    """
    Upload any video, audio, or image file.
    Returns verdict + per-modality confidence scores.

    Response JSON keys:
        verdict, confidence_pct, video_score, audio_score, image_score,
        metrics, audio_signature, model_data, gist
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTS)}"
            ),
        )

    # Write upload to a temp file so the inference engine can open it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = get_engine().predict(tmp_path)
        return JSONResponse(content=result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
