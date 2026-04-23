"""Dataset utilities for cryo-ET motor localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mrcfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.augmentations import build_augmentations
from data.preprocessing import (
    NormalizationConfig,
    build_spherical_target,
    extract_patch_3d,
    is_centroid_inside_patch,
    normalize_voxels,
    sample_hard_negative_center,
    world_to_patch_coords,
)


@dataclass
class SampleRecord:
    """Flat training sample metadata."""

    tomo_id: str
    tomo_path: str
    has_motor: int
    z: float
    y: float
    x: float


def _load_volume(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".mrc":
        with mrcfile.open(p, permissive=True) as mrc:
            return np.asarray(mrc.data, dtype=np.float32)
    if p.suffix.lower() == ".npy":
        return np.load(p).astype(np.float32)
    raise ValueError(f"Unsupported volume format: {path}")


class TomogramDataset(Dataset):
    """Patch-based 3D dataset with optional hard-negative mining."""

    def __init__(
        self,
        csv_path: str,
        patch_size: Sequence[int] = (96, 96, 96),
        hard_negative_ratio: float = 0.35,
        boundary_margin: int = 16,
        positive_radius: float = 6.0,
        augment_cfg: Optional[Dict[str, Any]] = None,
        normalization_cfg: Optional[Dict[str, Any]] = None,
        is_train: bool = True,
        seed: int = 42,
    ) -> None:
        self.records = self._read_records(csv_path)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.boundary_margin = int(boundary_margin)
        self.positive_radius = float(positive_radius)
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)
        self.norm_cfg = NormalizationConfig(**(normalization_cfg or {}))
        self.transforms = build_augmentations(augment_cfg or {}) if (is_train and augment_cfg and augment_cfg.get("enabled", True)) else None

        self.volume_cache: Dict[str, np.ndarray] = {}
        self.grouped = self._group_by_tomo(self.records)
        self.tomo_ids = list(self.grouped.keys())

    @staticmethod
    def _read_records(csv_path: str) -> List[SampleRecord]:
        df = pd.read_csv(csv_path)
        required = {"tomo_id", "tomo_path", "has_motor", "z", "y", "x"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
        return [
            SampleRecord(
                tomo_id=str(row.tomo_id),
                tomo_path=str(row.tomo_path),
                has_motor=int(row.has_motor),
                z=float(row.z),
                y=float(row.y),
                x=float(row.x),
            )
            for row in df.itertuples(index=False)
        ]

    @staticmethod
    def _group_by_tomo(records: List[SampleRecord]) -> Dict[str, List[SampleRecord]]:
        out: Dict[str, List[SampleRecord]] = {}
        for rec in records:
            out.setdefault(rec.tomo_id, []).append(rec)
        return out

    def _get_volume(self, tomo_path: str) -> np.ndarray:
        if tomo_path not in self.volume_cache:
            self.volume_cache[tomo_path] = normalize_voxels(_load_volume(tomo_path), self.norm_cfg)
        return self.volume_cache[tomo_path]

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _axis_bounds(size: int, patch: int) -> Tuple[int, int]:
        """Return inclusive valid center bounds for an axis.

        If volume axis is smaller than patch axis, return a single fallback center.
        """
        lo = patch // 2
        hi = size - (patch - patch // 2)
        if hi < lo:
            mid = max(0, size // 2)
            return mid, mid
        return lo, hi

    def _sample_random_center(self, volume_shape: Sequence[int]) -> Tuple[int, int, int]:
        center: List[int] = []
        for s, ps in zip(volume_shape, self.patch_size):
            lo, hi = self._axis_bounds(int(s), int(ps))
            if hi <= lo:
                center.append(int(lo))
            else:
                center.append(int(self.rng.integers(lo, hi + 1)))
        return tuple(center)  # type: ignore[return-value]

    def _clamp_center_to_volume(self, center: Sequence[int], volume_shape: Sequence[int]) -> Tuple[int, int, int]:
        out: List[int] = []
        for c, s, ps in zip(center, volume_shape, self.patch_size):
            lo, hi = self._axis_bounds(int(s), int(ps))
            out.append(int(np.clip(int(c), lo, hi)))
        return tuple(out)  # type: ignore[return-value]

    def _pick_patch_center(
        self, volume_shape: Sequence[int], motor_records: List[SampleRecord]
    ) -> Tuple[Tuple[int, int, int], int, List[float]]:
        if self.is_train:
            force_hard_negative = self.rng.random() < self.hard_negative_ratio
            if force_hard_negative:
                center = sample_hard_negative_center(
                    volume_shape=volume_shape,
                    patch_size=self.patch_size,
                    boundary_margin=self.boundary_margin,
                    rng=self.rng,
                )
                return center, 0, [0.0, 0.0, 0.0]

        if motor_records and self.rng.random() < 0.5:
            rec = motor_records[int(self.rng.integers(0, len(motor_records)))]
            center = (int(round(rec.z)), int(round(rec.y)), int(round(rec.x)))
            center = self._clamp_center_to_volume(center, volume_shape)
            local = world_to_patch_coords((rec.z, rec.y, rec.x), center, self.patch_size)
            return center, 1, local

        rand_center = self._sample_random_center(volume_shape)
        for rec in motor_records:
            if is_centroid_inside_patch((rec.z, rec.y, rec.x), rand_center, self.patch_size):
                local = world_to_patch_coords((rec.z, rec.y, rec.x), rand_center, self.patch_size)
                return rand_center, 1, local
        return rand_center, 0, [0.0, 0.0, 0.0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        rec = self.records[index]
        tomo_records = self.grouped[rec.tomo_id]
        motor_records = [r for r in tomo_records if r.has_motor == 1]
        volume = self._get_volume(rec.tomo_path)

        patch_center, label, centroid_local = self._pick_patch_center(volume.shape, motor_records)
        patch = extract_patch_3d(volume, center=patch_center, patch_size=self.patch_size)
        seg_target = (
            build_spherical_target(self.patch_size, centroid_local, self.positive_radius)
            if label == 1
            else np.zeros(self.patch_size, dtype=np.float32)
        )

        sample: Dict[str, Any] = {
            "image": patch[None, ...].astype(np.float32),
            "seg_target": seg_target[None, ...].astype(np.float32),
            "label": np.array([label], dtype=np.float32),
            "centroid": np.array(centroid_local, dtype=np.float32),
            "tomo_id": rec.tomo_id,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        else:
            sample = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in sample.items()}
        return sample
