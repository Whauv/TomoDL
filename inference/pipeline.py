"""Decoupled inference pipeline utilities with hybrid+uncertainty support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import mrcfile
import numpy as np
import pandas as pd
import torch

from core.errors import DataValidationError, InferenceError
from data.preprocessing import NormalizationConfig, normalize_voxels
from evaluation.tta import extract_centroid_from_heatmap, extract_instances_from_heatmap, tta_predict_heatmap
from models.detr3d import DETR3D
from models.hybrid_detector import HybridMotorDetector
from models.multitask_model import build_multitask_model
from models.resnet2d import ResNet2DSliceModel

EPSILON = 1e-8
SUBMISSION_COLUMNS = ["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]


def load_tomogram(path: str, norm_cfg: Dict[str, Any]) -> np.ndarray:
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


def _window_starts(size: int, patch: int, stride: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, max(1, size - patch + 1), max(1, stride)))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _extract_window_with_pad(volume: np.ndarray, start: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    z0, y0, x0 = start
    pz, py, px = patch_size
    z1, y1, x1 = z0 + pz, y0 + py, x0 + px
    crop = volume[z0:min(z1, volume.shape[0]), y0:min(y1, volume.shape[1]), x0:min(x1, volume.shape[2])]
    if crop.shape != patch_size:
        pad = [(0, pz - crop.shape[0]), (0, py - crop.shape[1]), (0, px - crop.shape[2])]
        crop = np.pad(crop, pad, mode="reflect")
    return crop


def _enable_dropout(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


class HeatmapPredictorFactory:
    """Factory for optional ensemble predictor components."""

    def __init__(self, cfg: Dict[str, Any], ckpt_path: str, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.ensemble_weights = cfg["evaluation"]["ensemble_weights"]
        self.inference_cfg = cfg.get("inference", {})
        self.unc_cfg = self.inference_cfg.get("uncertainty", {})
        self.use_hybrid = bool(self.inference_cfg.get("use_hybrid_detector", False))
        patch_size = tuple(cfg["data"]["patch_size"])

        if self.use_hybrid:
            self.model3d = HybridMotorDetector(cfg["model"], patch_size=patch_size).to(device)
        else:
            self.model3d = build_multitask_model(cfg["model"], patch_size=patch_size).to(device)

        state = torch.load(ckpt_path, map_location="cpu")
        self.model3d.load_state_dict(state.get("model", state), strict=False)
        self.model3d.eval()

        self.model2d = None
        if float(self.ensemble_weights.get("resnet2d", 0.0)) > 0.0:
            self.model2d = ResNet2DSliceModel(in_channels=1, pretrained=False).to(device).eval()

        self.modeldetr = None
        if float(self.ensemble_weights.get("detr3d", 0.0)) > 0.0:
            self.modeldetr = DETR3D(in_channels=1).to(device).eval()

    def _single_pass(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        w3d = float(self.ensemble_weights.get("unet3d", 0.0))
        w2d = float(self.ensemble_weights.get("resnet2d", 0.0))
        wdetr = float(self.ensemble_weights.get("detr3d", 0.0))
        total_w = w3d + w2d + wdetr
        if total_w <= 0.0:
            raise InferenceError("At least one ensemble weight must be greater than zero.")

        if bool(self.cfg["evaluation"]["tta_rotations"]):
            policy = str(self.cfg.get("evaluation", {}).get("tta_policy", "full"))
            heat3d = tta_predict_heatmap(self.model3d, x, policy=policy)
        else:
            raw = self.model3d(x)
            if self.use_hybrid and "fused_heatmap" in raw:
                heat3d = raw["fused_heatmap"]
            else:
                heat3d = torch.sigmoid(raw["segmentation"])

        heat2d = torch.sigmoid(self.model2d(x)["heatmap"]) if self.model2d is not None else torch.zeros_like(heat3d)
        heatd = torch.sigmoid(self.modeldetr(x)["heatmap"]) if self.modeldetr is not None else torch.zeros_like(heat3d)
        heat = (w3d * heat3d + w2d * heat2d + wdetr * heatd) / (total_w + EPSILON)

        q_unc = 0.0
        if self.use_hybrid:
            raw = self.model3d(x)
            if "query_uncertainty" in raw:
                q_unc = float(raw["query_uncertainty"].mean().detach().item())
        return heat, q_unc

    def predict_heatmap_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        method = str(self.unc_cfg.get("method", "none")).lower()
        passes = int(self.unc_cfg.get("mc_passes", 1))
        if method != "mc_dropout" or passes <= 1:
            with torch.no_grad():
                heat, q_unc = self._single_pass(x)
            return heat, float(q_unc)

        self.model3d.train()
        _enable_dropout(self.model3d)
        heats = []
        q_uncs = []
        with torch.no_grad():
            for _ in range(passes):
                h, q = self._single_pass(x)
                heats.append(h)
                q_uncs.append(q)
        heat_stack = torch.stack(heats, dim=0)
        mean_heat = heat_stack.mean(dim=0)
        epi_unc = float(heat_stack.var(dim=0).mean().item())
        query_unc = float(np.mean(q_uncs)) if q_uncs else 0.0
        self.model3d.eval()
        return mean_heat, epi_unc + query_unc


def load_inference_manifest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"tomo_id", "tomo_path"}
    if not required.issubset(df.columns):
        raise DataValidationError(f"test_csv must contain columns {sorted(required)}")
    return df


def _predict_peak_coord_unc(
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    predictor: HeatmapPredictorFactory,
    device: torch.device,
    cfg: Dict[str, Any],
    sliding_window: bool,
    window_overlap: float,
    low_memory_mode: bool,
) -> tuple[float, tuple[float, float, float], float]:
    if not sliding_window:
        centered = center_crop_or_pad_3d(volume, patch_size)
        x = torch.from_numpy(centered[None, None, ...]).float().to(device)
        heat, unc = predictor.predict_heatmap_with_uncertainty(x)
        peak_score = float(heat.max().item())
        inf = cfg.get("inference", {})
        instances = extract_instances_from_heatmap(
            heat,
            threshold=float(inf.get("instance_threshold", 0.5)),
            min_size=int(inf.get("instance_min_size", 8)),
            nms_distance=int(inf.get("instance_nms_distance", 6)),
        )
        centroid = instances[0]["centroid"] if instances else extract_centroid_from_heatmap(heat)[0].cpu().numpy().tolist()
        if low_memory_mode and device.type == "cuda":
            del x, heat
            torch.cuda.empty_cache()
        return peak_score, (float(centroid[0]), float(centroid[1]), float(centroid[2])), float(unc)

    pz, py, px = patch_size
    stride = (
        max(1, int(round(pz * (1.0 - window_overlap)))),
        max(1, int(round(py * (1.0 - window_overlap)))),
        max(1, int(round(px * (1.0 - window_overlap)))),
    )
    z_starts = _window_starts(volume.shape[0], pz, stride[0])
    y_starts = _window_starts(volume.shape[1], py, stride[1])
    x_starts = _window_starts(volume.shape[2], px, stride[2])

    best_score = -1.0
    best_coord = (0.0, 0.0, 0.0)
    best_unc = 0.0
    for zs in z_starts:
        for ys in y_starts:
            for xs in x_starts:
                patch = _extract_window_with_pad(volume, (zs, ys, xs), patch_size)
                x = torch.from_numpy(patch[None, None, ...]).float().to(device)
                heat, unc = predictor.predict_heatmap_with_uncertainty(x)
                flat = heat.flatten(1)
                idx = int(flat.argmax(dim=1).item())
                score = float(flat.max().item())
                if score > best_score:
                    ly = (idx % (py * px)) // px
                    lx = idx % px
                    lz = idx // (py * px)
                    best_score = score
                    best_coord = (float(zs + lz), float(ys + ly), float(xs + lx))
                    best_unc = float(unc)
                if low_memory_mode and device.type == "cuda":
                    del x, heat
                    torch.cuda.empty_cache()
    return best_score, best_coord, best_unc


def predict_submission_rows(test_df: pd.DataFrame, cfg: Dict[str, Any], predictor: HeatmapPredictorFactory, device: torch.device) -> list[dict[str, float | str]]:
    patch_size = tuple(cfg["data"]["patch_size"])
    inf_cfg = cfg.get("inference", {})
    unc_cfg = inf_cfg.get("uncertainty", {})
    no_motor_threshold = float(inf_cfg.get("no_motor_threshold", cfg.get("evaluation", {}).get("decision_threshold", 0.5)))
    unc_threshold = float(unc_cfg.get("threshold", 1e9))
    low_memory_mode = bool(inf_cfg.get("low_memory_mode", False))
    sliding_window = bool(inf_cfg.get("sliding_window_if_large", False))
    window_overlap = float(inf_cfg.get("window_overlap", 0.25))

    rows: list[dict[str, float | str]] = []
    for row in test_df.itertuples(index=False):
        volume = load_tomogram(str(row.tomo_path), cfg["data"]["normalization"])
        run_sliding = sliding_window and any(vs > ps for vs, ps in zip(volume.shape, patch_size))
        peak_score, centroid, unc = _predict_peak_coord_unc(
            volume=volume,
            patch_size=patch_size,
            predictor=predictor,
            device=device,
            cfg=cfg,
            sliding_window=run_sliding,
            window_overlap=window_overlap,
            low_memory_mode=low_memory_mode,
        )

        if (peak_score < no_motor_threshold) or (unc > unc_threshold):
            rows.append({"tomo_id": str(row.tomo_id), "Motor axis 0": -1.0, "Motor axis 1": -1.0, "Motor axis 2": -1.0})
        else:
            rows.append({"tomo_id": str(row.tomo_id), "Motor axis 0": float(centroid[0]), "Motor axis 1": float(centroid[1]), "Motor axis 2": float(centroid[2])})
    return rows
