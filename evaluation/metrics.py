"""Evaluation metrics for motor detection/localization."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def f2_score_from_counts(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    """Compute F-beta score from count statistics."""
    beta2 = beta * beta
    num = (1 + beta2) * tp
    den = (1 + beta2) * tp + beta2 * fn + fp
    return float(num / den) if den > 0 else 0.0


def localization_match(
    pred: Sequence[float],
    target: Sequence[float],
    tolerance: float = 10.0,
) -> bool:
    """True if Euclidean centroid distance is within tolerance."""
    pred_np = np.asarray(pred, dtype=np.float32)
    tgt_np = np.asarray(target, dtype=np.float32)
    return float(np.linalg.norm(pred_np - tgt_np)) <= float(tolerance)


def evaluate_batch_localization(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    labels: torch.Tensor,
    tolerance: float,
) -> Tuple[int, int, int]:
    """Compute TP/FP/FN at batch level based on centroid tolerance and presence labels."""
    pred_coords = pred_coords.detach().cpu()
    true_coords = true_coords.detach().cpu()
    labels = labels.detach().cpu().view(-1)
    tp = fp = fn = 0
    for i in range(pred_coords.shape[0]):
        has_motor = int(labels[i].item() > 0.5)
        pred = pred_coords[i].tolist()
        tgt = true_coords[i].tolist()
        if has_motor == 1 and localization_match(pred, tgt, tolerance):
            tp += 1
        elif has_motor == 1:
            fn += 1
        else:
            fp += 1 if np.linalg.norm(np.asarray(pred)) > 1e-3 else 0
    return tp, fp, fn
