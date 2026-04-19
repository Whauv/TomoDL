"""Preprocessing utilities for cryo-ET tomograms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class NormalizationConfig:
    """Configuration for voxel intensity normalization."""

    clip_min: float = -4.0
    clip_max: float = 4.0


def normalize_voxels(volume: np.ndarray, cfg: NormalizationConfig) -> np.ndarray:
    """Normalize a 3D volume with robust z-score followed by clipping."""
    volume = volume.astype(np.float32, copy=False)
    mean = float(volume.mean())
    std = float(volume.std()) + 1e-6
    norm = (volume - mean) / std
    return np.clip(norm, cfg.clip_min, cfg.clip_max)


def extract_patch_3d(
    volume: np.ndarray,
    center: Sequence[int],
    patch_size: Sequence[int],
) -> np.ndarray:
    """Extract a 3D patch centered at `center` with reflection padding at boundaries."""
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    cz, cy, cx = center

    z0, z1 = cz - pd // 2, cz + (pd - pd // 2)
    y0, y1 = cy - ph // 2, cy + (ph - ph // 2)
    x0, x1 = cx - pw // 2, cx + (pw - pw // 2)

    pad_before = [max(0, -z0), max(0, -y0), max(0, -x0)]
    pad_after = [max(0, z1 - d), max(0, y1 - h), max(0, x1 - w)]
    if any(v > 0 for v in pad_before + pad_after):
        volume = np.pad(
            volume,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="reflect",
        )
        z0 += pad_before[0]
        z1 += pad_before[0]
        y0 += pad_before[1]
        y1 += pad_before[1]
        x0 += pad_before[2]
        x1 += pad_before[2]

    return volume[z0:z1, y0:y1, x0:x1]


def build_spherical_target(
    patch_size: Sequence[int],
    centroid: Sequence[float],
    radius: float,
) -> np.ndarray:
    """Create a binary sphere mask centered at `centroid` in patch coordinates."""
    zz, yy, xx = np.indices(patch_size, dtype=np.float32)
    cz, cy, cx = centroid
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return (dist2 <= radius**2).astype(np.float32)


def sample_hard_negative_center(
    volume_shape: Sequence[int],
    patch_size: Sequence[int],
    boundary_margin: int,
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    """Sample a center close to tomogram boundaries for hard negatives."""
    d, h, w = volume_shape
    pd, ph, pw = patch_size
    mins = [pd // 2, ph // 2, pw // 2]
    maxs = [d - (pd - pd // 2), h - (ph - ph // 2), w - (pw - pw // 2)]

    axis = int(rng.integers(0, 3))
    side = int(rng.integers(0, 2))
    center = []
    for i, (mn, mx) in enumerate(zip(mins, maxs)):
        if i == axis:
            if side == 0:
                lo, hi = mn, min(mx, mn + boundary_margin)
            else:
                lo, hi = max(mn, mx - boundary_margin), mx
        else:
            lo, hi = mn, mx
        if hi <= lo:
            center.append(int((mn + mx) // 2))
        else:
            center.append(int(rng.integers(lo, hi + 1)))
    return tuple(center)  # type: ignore[return-value]


def is_centroid_inside_patch(
    centroid_world: Sequence[float],
    patch_center_world: Sequence[int],
    patch_size: Sequence[int],
) -> bool:
    """Return whether world-space centroid lies inside patch bounds."""
    for c, pc, ps in zip(centroid_world, patch_center_world, patch_size):
        lo = pc - ps / 2.0
        hi = pc + ps / 2.0
        if c < lo or c >= hi:
            return False
    return True


def world_to_patch_coords(
    centroid_world: Sequence[float],
    patch_center_world: Sequence[int],
    patch_size: Sequence[int],
) -> List[float]:
    """Convert world coordinate centroid to patch-local coordinates."""
    coords: List[float] = []
    for c, pc, ps in zip(centroid_world, patch_center_world, patch_size):
        coords.append(float(c - (pc - ps / 2.0)))
    return coords
