"""
metrics.py — Evaluation metrics for DeepGuard / Sach-AI.

Usage:
    from utils.metrics import compute_metrics
    metrics = compute_metrics(labels, preds, probs)
"""

from typing import Union
import numpy as np


def compute_metrics(
    labels: list,
    preds:  list,
    probs:  list,
) -> dict:
    """
    Compute classification metrics for binary deepfake detection.

    Args:
        labels : ground-truth binary labels (0 = real, 1 = fake)
        preds  : predicted binary labels at threshold 0.5
        probs  : predicted probabilities P(fake)

    Returns:
        dict with keys: accuracy, precision, recall, f1, auc
    """
    labels = np.array(labels).ravel().astype(int)
    preds  = np.array(preds).ravel().astype(int)
    probs  = np.array(probs).ravel().astype(float)

    n = len(labels)
    if n == 0:
        return dict(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, auc=0.5)

    # ── Basic counts ──────────────────────────────────────────────────────────
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    accuracy  = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # ── AUC-ROC (pure NumPy — no sklearn dependency required) ─────────────────
    auc = _roc_auc(labels, probs)

    return dict(
        accuracy  = round(accuracy,  4),
        precision = round(precision, 4),
        recall    = round(recall,    4),
        f1        = round(f1,        4),
        auc       = round(auc,       4),
    )


def _roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute ROC-AUC using the rank-based Wilcoxon-Mann-Whitney formula.
    This handles tied probabilities correctly and is more robust than
    simple trapezoidal integration.
    """
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Use scipy style ranking if possible, otherwise manual
    # To avoid external dependencies, we use a pure numpy approach:
    # Sort indices by probability
    indices = np.argsort(probs)
    labels  = labels[indices]
    probs   = probs[indices]

    # Handle ties by assigning mid-ranks
    ranks = np.zeros_like(probs)
    i = 0
    while i < len(probs):
        j = i + 1
        while j < len(probs) and probs[j] == probs[i]:
            j += 1
        # Mid-rank for the block [i, j)
        ranks[i:j] = (i + j + 1) / 2.0
        i = j

    # AUC = (Sum of ranks of positive samples - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    pos_rank_sum = np.sum(ranks[labels == 1])
    auc = (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)

    return float(auc)
