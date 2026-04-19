"""Ensembling utilities for 3D U-Net + 2D ResNet + DETR-3D."""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Tuple

import torch

from evaluation.metrics import f2_score_from_counts
from evaluation.tta import extract_centroid_from_heatmap


def weighted_heatmap_ensemble(
    heat_unet3d: torch.Tensor,
    heat_resnet2d: torch.Tensor,
    heat_detr3d: torch.Tensor,
    weights: Dict[str, float],
) -> torch.Tensor:
    """Weighted average of three heatmaps."""
    w1, w2, w3 = weights["unet3d"], weights["resnet2d"], weights["detr3d"]
    denom = w1 + w2 + w3 + 1e-8
    return (w1 * heat_unet3d + w2 * heat_resnet2d + w3 * heat_detr3d) / denom


def optimize_ensemble_weights(
    val_predictions: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    tolerance: float = 10.0,
) -> Dict[str, float]:
    """Grid-search ensemble weights based on validation F2."""
    best_w = {"unet3d": 0.5, "resnet2d": 0.25, "detr3d": 0.25}
    best_f2 = -1.0
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for w1, w2, w3 in product(grid, grid, grid):
        if abs((w1 + w2 + w3) - 1.0) > 1e-5:
            continue
        tp = fp = fn = 0
        for h1, h2, h3, tgt_centroid, labels in val_predictions:
            heat = weighted_heatmap_ensemble(h1, h2, h3, {"unet3d": w1, "resnet2d": w2, "detr3d": w3})
            pred = extract_centroid_from_heatmap(heat).cpu()
            tgt = tgt_centroid.cpu()
            lbl = labels.cpu().view(-1)
            for i in range(pred.shape[0]):
                if int(lbl[i].item() > 0.5) == 1:
                    dist = torch.norm(pred[i] - tgt[i]).item()
                    if dist <= tolerance:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if torch.norm(pred[i]).item() > 1e-3:
                        fp += 1
        f2 = f2_score_from_counts(tp, fp, fn)
        if f2 > best_f2:
            best_f2 = f2
            best_w = {"unet3d": w1, "resnet2d": w2, "detr3d": w3}
    return best_w
