"""3D augmentation pipeline using MONAI transforms."""

from __future__ import annotations

from typing import Any, Dict

from monai.transforms import (
    Compose,
    EnsureTyped,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
)


def build_augmentations(cfg: Dict[str, Any]) -> Compose:
    """Build 3D dictionary-based augmentations for image + seg_target."""
    return Compose(
        [
            Rand3DElasticd(
                keys=["image", "seg_target"],
                sigma_range=(3.0, 6.0),
                magnitude_range=(20.0, 60.0),
                prob=float(cfg.get("elastic_prob", 0.2)),
                rotate_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandFlipd(keys=["image", "seg_target"], prob=float(cfg.get("flip_prob", 0.5)), spatial_axis=0),
            RandFlipd(keys=["image", "seg_target"], prob=float(cfg.get("flip_prob", 0.5)), spatial_axis=1),
            RandFlipd(keys=["image", "seg_target"], prob=float(cfg.get("flip_prob", 0.5)), spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=float(cfg.get("noise_prob", 0.2)), mean=0.0, std=0.08),
            RandAdjustContrastd(keys=["image"], prob=float(cfg.get("contrast_prob", 0.2)), gamma=(0.7, 1.5)),
            EnsureTyped(keys=["image", "seg_target", "label", "centroid"]),
        ]
    )
