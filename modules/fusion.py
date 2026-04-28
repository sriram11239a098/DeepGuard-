"""
fusion.py — Legacy weighted-average fusion utility.

This module provides the `fuse()` helper used when individual branch scores
are available without running the full neural fusion model (e.g. in testing
or when only some modalities are present).

The trained MultimodalFusionModel in fusion_model.py learns its own fusion
weights end-to-end — this file is a complementary utility, not a replacement.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


@dataclass
class BranchResult:
    """Prediction from a single detection branch."""
    score:   Optional[float] = None   # P(fake) ∈ [0, 1], or None if unavailable
    verdict: Optional[str]   = None   # "DEEPFAKE" | "AUTHENTIC" | None


@dataclass
class FusionResult:
    """Aggregated output from the weighted fusion layer."""
    video:          BranchResult = field(default_factory=BranchResult)
    audio:          BranchResult = field(default_factory=BranchResult)
    image:          BranchResult = field(default_factory=BranchResult)
    final_score:    float        = 0.0
    verdict:        str          = "UNKNOWN"
    confidence_pct: float        = 0.0
    timestamp:      str          = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "verdict":        self.verdict,
            "confidence":     round(self.final_score, 4),
            "confidence_pct": round(self.confidence_pct, 2),
            "video_score":    self.video.score,
            "audio_score":    self.audio.score,
            "image_score":    self.image.score,
            "video_verdict":  self.video.verdict,
            "audio_verdict":  self.audio.verdict,
            "image_verdict":  self.image.verdict,
            "timestamp":      self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _branch_result(score: Optional[float]) -> BranchResult:
    if score is None:
        return BranchResult(score=None, verdict=None)
    verdict = "DEEPFAKE" if score >= config.DETECTION_THRESHOLD else "AUTHENTIC"
    return BranchResult(score=round(score, 4), verdict=verdict)


def fuse(
    video_score: Optional[float] = None,
    audio_score: Optional[float] = None,
    image_score: Optional[float] = None,
) -> FusionResult:
    """
    Weighted fusion of per-modality deepfake probabilities.

    Available modalities are re-normalised so weights still sum to 1
    even when some branches are absent.

    Args:
        video_score : P(fake) from video branch, or None
        audio_score : P(fake) from audio branch, or None
        image_score : P(fake) from image branch, or None

    Returns:
        FusionResult with final verdict and per-branch details.
    """
    result = FusionResult(
        video=_branch_result(video_score),
        audio=_branch_result(audio_score),
        image=_branch_result(image_score),
    )

    pairs = []
    if video_score is not None:
        pairs.append((video_score, config.FUSION_WEIGHT_VIDEO))
    if audio_score is not None:
        pairs.append((audio_score, config.FUSION_WEIGHT_AUDIO))
    if image_score is not None:
        pairs.append((image_score, config.FUSION_WEIGHT_IMAGE))

    if not pairs:
        result.verdict = "UNKNOWN"
        return result

    total_weight = sum(w for _, w in pairs)
    final_score  = sum(s * w for s, w in pairs) / total_weight

    result.final_score    = round(final_score, 4)
    result.confidence_pct = round(final_score * 100, 2)
    result.verdict        = (
        "DEEPFAKE" if final_score >= config.DETECTION_THRESHOLD else "AUTHENTIC"
    )
    return result
