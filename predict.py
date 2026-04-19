"""Inference script for Kaggle submission generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml

from data.preprocessing import NormalizationConfig, normalize_voxels
from evaluation.tta import extract_centroid_from_heatmap, tta_predict_heatmap
from models.detr3d import DETR3D
from models.multitask_model import build_multitask_model
from models.resnet2d import ResNet2DSliceModel


def load_config(path: str) -> Dict[str, Any]:
    """Load project config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_volume(path: str, norm_cfg: Dict[str, Any]) -> np.ndarray:
    """Load .mrc/.npy tomogram and normalize."""
    p = Path(path)
    if p.suffix.lower() == ".mrc":
        with mrcfile.open(p, permissive=True) as mrc:
            vol = np.asarray(mrc.data, dtype=np.float32)
    elif p.suffix.lower() == ".npy":
        vol = np.load(p).astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {path}")
    return normalize_voxels(vol, NormalizationConfig(**norm_cfg))


def center_crop_or_pad(volume: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or reflect-pad volume to patch size."""
    out = volume
    for axis, size in enumerate(patch_size):
        cur = out.shape[axis]
        if cur < size:
            before = (size - cur) // 2
            after = size - cur - before
            pads = [(0, 0), (0, 0), (0, 0)]
            pads[axis] = (before, after)
            out = np.pad(out, pads, mode="reflect")
        elif cur > size:
            start = (cur - size) // 2
            end = start + size
            slicer = [slice(None), slice(None), slice(None)]
            slicer[axis] = slice(start, end)
            out = out[tuple(slicer)]
    return out


def run_inference(cfg: Dict[str, Any], ckpt_path: str, device: torch.device) -> pd.DataFrame:
    """Generate centroid predictions and return Kaggle submission dataframe."""
    test_df = pd.read_csv(cfg["paths"]["test_csv"])
    required = {"tomo_id", "tomo_path"}
    if not required.issubset(test_df.columns):
        raise ValueError(f"test_csv must contain columns {sorted(required)}")

    model = build_multitask_model(cfg["model"], patch_size=tuple(cfg["data"]["patch_size"])).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    res2d = ResNet2DSliceModel(in_channels=1, pretrained=False).to(device).eval()
    detr = DETR3D(in_channels=1).to(device).eval()
    w = cfg["evaluation"]["ensemble_weights"]

    rows: List[Dict[str, float | str]] = []
    for row in test_df.itertuples(index=False):
        vol = load_volume(str(row.tomo_path), cfg["data"]["normalization"])
        vol = center_crop_or_pad(vol, tuple(cfg["data"]["patch_size"]))
        x = torch.from_numpy(vol[None, None, ...]).float().to(device)

        with torch.no_grad():
            if bool(cfg["evaluation"]["tta_rotations"]):
                heat3d = tta_predict_heatmap(model, x)
            else:
                heat3d = torch.sigmoid(model(x)["segmentation"])
            heat2d = torch.sigmoid(res2d(x)["heatmap"])
            heatd = torch.sigmoid(detr(x)["heatmap"])
            heat = (w["unet3d"] * heat3d + w["resnet2d"] * heat2d + w["detr3d"] * heatd) / (
                w["unet3d"] + w["resnet2d"] + w["detr3d"] + 1e-8
            )
            centroid = extract_centroid_from_heatmap(heat)[0].cpu().numpy()
        rows.append({"tomo_id": str(row.tomo_id), "x": float(centroid[2]), "y": float(centroid[1]), "z": float(centroid[0])})

    return pd.DataFrame(rows, columns=["tomo_id", "x", "y", "z"])


def main() -> None:
    """CLI entrypoint for submission generation."""
    parser = argparse.ArgumentParser(description="TomoDL inference")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt = args.checkpoint or cfg["paths"]["finetune_ckpt"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    submission = run_inference(cfg, ckpt, device)
    out_path = Path(cfg["paths"]["submission_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
