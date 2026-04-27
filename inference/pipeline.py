"""Decoupled inference pipeline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import mrcfile
import numpy as np
import pandas as pd
import torch

from core.errors import DataValidationError, InferenceError
from data.preprocessing import NormalizationConfig, normalize_voxels
from evaluation.tta import extract_centroid_from_heatmap, tta_predict_heatmap
from models.detr3d import DETR3D
from models.multitask_model import build_multitask_model
from models.resnet2d import ResNet2DSliceModel

EPSILON = 1e-8
SUBMISSION_COLUMNS = ["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]


def load_tomogram(path: str, norm_cfg: Dict[str, Any]) -> np.ndarray:
    """Load a tomogram file and normalize voxel intensities."""
    p = Path(path)
    if p.suffix.lower() == ".mrc":
        with mrcfile.open(p, permissive=True) as mrc:
            volume = np.asarray(mrc.data, dtype=np.float32)
    elif p.suffix.lower() == ".npy":
        volume = np.load(p).astype(np.float32)
    else:
        raise DataValidationError(f"Unsupported tomogram format: {path}")
    return normalize_voxels(volume, NormalizationConfig(**norm_cfg))


def center_crop_or_pad_3d(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or reflect-pad a 3D volume to target shape."""
    out = volume
    for axis, target in enumerate(target_shape):
        current = out.shape[axis]
        if current < target:
            pre = (target - current) // 2
            post = target - current - pre
            pads = [(0, 0), (0, 0), (0, 0)]
            pads[axis] = (pre, post)
            out = np.pad(out, pads, mode="reflect")
        elif current > target:
            start = (current - target) // 2
            end = start + target
            slicer = [slice(None), slice(None), slice(None)]
            slicer[axis] = slice(start, end)
            out = out[tuple(slicer)]
    return out


class HeatmapPredictorFactory:
    """Factory for optional ensemble predictor components."""

    def __init__(self, cfg: Dict[str, Any], ckpt_path: str, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.ensemble_weights = cfg["evaluation"]["ensemble_weights"]
        patch_size = tuple(cfg["data"]["patch_size"])

        self.model3d = build_multitask_model(cfg["model"], patch_size=patch_size).to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        self.model3d.load_state_dict(state["model"], strict=False)
        self.model3d.eval()

        self.model2d = None
        if float(self.ensemble_weights.get("resnet2d", 0.0)) > 0.0:
            self.model2d = ResNet2DSliceModel(in_channels=1, pretrained=False).to(device).eval()

        self.modeldetr = None
        if float(self.ensemble_weights.get("detr3d", 0.0)) > 0.0:
            self.modeldetr = DETR3D(in_channels=1).to(device).eval()

    def predict_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        """Generate final weighted heatmap."""
        w3d = float(self.ensemble_weights.get("unet3d", 0.0))
        w2d = float(self.ensemble_weights.get("resnet2d", 0.0))
        wdetr = float(self.ensemble_weights.get("detr3d", 0.0))
        total_w = w3d + w2d + wdetr
        if total_w <= 0.0:
            raise InferenceError("At least one ensemble weight must be greater than zero.")

        with torch.no_grad():
            if bool(self.cfg["evaluation"]["tta_rotations"]):
                heat3d = tta_predict_heatmap(self.model3d, x)
            else:
                heat3d = torch.sigmoid(self.model3d(x)["segmentation"])

            if self.model2d is not None:
                heat2d = torch.sigmoid(self.model2d(x)["heatmap"])
            else:
                heat2d = torch.zeros_like(heat3d)

            if self.modeldetr is not None:
                heatd = torch.sigmoid(self.modeldetr(x)["heatmap"])
            else:
                heatd = torch.zeros_like(heat3d)

            heat = (w3d * heat3d + w2d * heat2d + wdetr * heatd) / (total_w + EPSILON)
            return heat


def load_inference_manifest(csv_path: str) -> pd.DataFrame:
    """Read and validate inference manifest."""
    df = pd.read_csv(csv_path)
    required = {"tomo_id", "tomo_path"}
    if not required.issubset(df.columns):
        raise DataValidationError(f"test_csv must contain columns {sorted(required)}")
    return df


def predict_submission_rows(
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    predictor: HeatmapPredictorFactory,
    device: torch.device,
) -> list[dict[str, float | str]]:
    """Predict rows for submission from a validated test manifest."""
    patch_size = tuple(cfg["data"]["patch_size"])
    no_motor_threshold = float(
        cfg.get("inference", {}).get(
            "no_motor_threshold",
            cfg.get("evaluation", {}).get("decision_threshold", 0.5),
        )
    )
    rows: list[dict[str, float | str]] = []
    for row in test_df.itertuples(index=False):
        volume = load_tomogram(str(row.tomo_path), cfg["data"]["normalization"])
        volume = center_crop_or_pad_3d(volume, patch_size)
        x = torch.from_numpy(volume[None, None, ...]).float().to(device)
        heat = predictor.predict_heatmap(x)
        peak_score = float(heat.max().item())

        if peak_score < no_motor_threshold:
            rows.append(
                {
                    "tomo_id": str(row.tomo_id),
                    "Motor axis 0": -1.0,
                    "Motor axis 1": -1.0,
                    "Motor axis 2": -1.0,
                }
            )
        else:
            centroid = extract_centroid_from_heatmap(heat)[0].cpu().numpy()
            rows.append(
                {
                    "tomo_id": str(row.tomo_id),
                    # Competition expects axes in tomogram order.
                    "Motor axis 0": float(centroid[0]),
                    "Motor axis 1": float(centroid[1]),
                    "Motor axis 2": float(centroid[2]),
                }
            )
    return rows
