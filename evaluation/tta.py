"""Test-time augmentation with 3D rotational variants."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


ROTATIONS: List[Tuple[Tuple[int, int], int]] = [
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), 2),
    ((2, 3), 3),
    ((2, 4), 1),
    ((2, 4), 2),
    ((3, 4), 1),
    ((3, 4), 2),
]


def _rotate(x: torch.Tensor, dims: Tuple[int, int], k: int) -> torch.Tensor:
    return torch.rot90(x, k=k, dims=dims) if k > 0 else x


def tta_predict_heatmap(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply 8 rotational variants and average inverse-rotated heatmaps."""
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for dims, k in ROTATIONS:
            x_rot = _rotate(x, dims=dims, k=k)
            out = model(x_rot)
            heat = torch.sigmoid(out["segmentation"])
            heat_inv = _rotate(heat, dims=dims, k=(4 - k) % 4)
            preds.append(heat_inv)
    return torch.stack(preds, dim=0).mean(dim=0)


def extract_centroid_from_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """Extract centroid from heatmap via argmax in (z, y, x) order."""
    b = heatmap.shape[0]
    coords = []
    flat = heatmap[:, 0].flatten(1)
    idx = flat.argmax(dim=1)
    d, h, w = heatmap.shape[2:]
    z = idx // (h * w)
    y = (idx % (h * w)) // w
    x = idx % w
    for i in range(b):
        coords.append(torch.stack([z[i], y[i], x[i]]).float())
    return torch.stack(coords, dim=0).to(heatmap.device)
