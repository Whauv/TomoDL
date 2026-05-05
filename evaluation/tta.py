"""Test-time augmentation and multi-instance extraction for 3D heatmaps."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

FULL_ROTATIONS: List[Tuple[Tuple[int, int], int]] = [
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), 2),
    ((2, 3), 3),
    ((2, 4), 1),
    ((2, 4), 2),
    ((3, 4), 1),
    ((3, 4), 2),
]

# High-value rotations selected from ablation-friendly subset.
HIGH_VALUE_ROTATIONS: List[Tuple[Tuple[int, int], int]] = [
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 4), 1),
    ((3, 4), 1),
]


def _rotate(x: torch.Tensor, dims: Tuple[int, int], k: int) -> torch.Tensor:
    return torch.rot90(x, k=k, dims=dims) if k > 0 else x


def _select_rotations(policy: str) -> List[Tuple[Tuple[int, int], int]]:
    key = str(policy).lower()
    if key in {"high_value", "lite"}:
        return HIGH_VALUE_ROTATIONS
    if key in {"none", "off"}:
        return [((2, 3), 0)]
    return FULL_ROTATIONS


def tta_predict_heatmap(model: torch.nn.Module, x: torch.Tensor, policy: str = "full") -> torch.Tensor:
    """Apply rotational TTA variants and average inverse-rotated heatmaps."""
    preds: List[torch.Tensor] = []
    rotations = _select_rotations(policy)
    with torch.no_grad():
        for dims, k in rotations:
            x_rot = _rotate(x, dims=dims, k=k)
            out = model(x_rot)
            heat = torch.sigmoid(out["segmentation"])
            heat_inv = _rotate(heat, dims=dims, k=(4 - k) % 4)
            preds.append(heat_inv)
    return torch.stack(preds, dim=0).mean(dim=0)


def extract_centroid_from_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """Extract single best centroid from heatmap via argmax in (z, y, x) order."""
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


def _nms_points(points: List[tuple[int, int, int, float]], min_distance: int) -> List[tuple[int, int, int, float]]:
    kept: List[tuple[int, int, int, float]] = []
    for p in sorted(points, key=lambda t: t[3], reverse=True):
        z, y, x, s = p
        ok = True
        for kz, ky, kx, _ in kept:
            if (z - kz) ** 2 + (y - ky) ** 2 + (x - kx) ** 2 < (min_distance ** 2):
                ok = False
                break
        if ok:
            kept.append(p)
    return kept


def extract_instances_from_heatmap(
    heatmap: torch.Tensor,
    threshold: float = 0.5,
    min_size: int = 8,
    nms_distance: int = 6,
) -> List[Dict[str, float | list[float]]]:
    """Connected-components extraction for candidate-set motor prediction.

    Returns list of dicts with centroid [z,y,x] and score.
    """
    if heatmap.ndim != 5:
        raise ValueError("Expected heatmap shape [B, 1, D, H, W].")
    arr = heatmap[0, 0].detach().cpu().numpy().astype(np.float32)
    mask = arr >= float(threshold)
    if not mask.any():
        return []

    d, h, w = arr.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[List[tuple[int, int, int]]] = []
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    coords = np.argwhere(mask)
    for z0, y0, x0 in coords:
        if visited[z0, y0, x0]:
            continue
        q = deque([(int(z0), int(y0), int(x0))])
        visited[z0, y0, x0] = True
        comp: List[tuple[int, int, int]] = []
        while q:
            z, y, x = q.popleft()
            comp.append((z, y, x))
            for dz, dy, dx in neighbors:
                nz, ny, nx = z + dz, y + dy, x + dx
                if nz < 0 or ny < 0 or nx < 0 or nz >= d or ny >= h or nx >= w:
                    continue
                if visited[nz, ny, nx] or not mask[nz, ny, nx]:
                    continue
                visited[nz, ny, nx] = True
                q.append((nz, ny, nx))
        if len(comp) >= int(min_size):
            components.append(comp)

    candidates: List[tuple[int, int, int, float]] = []
    for comp in components:
        comp_np = np.asarray(comp, dtype=np.int32)
        scores = arr[comp_np[:, 0], comp_np[:, 1], comp_np[:, 2]]
        peak_idx = int(np.argmax(scores))
        pz, py, px = comp_np[peak_idx].tolist()
        score = float(scores[peak_idx])
        candidates.append((int(pz), int(py), int(px), score))

    kept = _nms_points(candidates, min_distance=int(nms_distance))
    out: List[Dict[str, float | list[float]]] = []
    for z, y, x, s in kept:
        out.append({"centroid": [float(z), float(y), float(x)], "score": float(s)})
    return out
