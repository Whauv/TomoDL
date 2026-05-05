"""Calibrate no-motor threshold using validation peaks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

from core.config import load_yaml_config
from core.runtime import resolve_device
from evaluation.metrics import f2_score_from_counts
from inference.pipeline import HeatmapPredictorFactory, load_tomogram


def _counts_from_threshold(peaks: np.ndarray, labels: np.ndarray, thr: float) -> tuple[int, int, int]:
    pred = peaks >= thr
    gt = labels.astype(bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt)))
    fn = int(np.sum((~pred) & gt))
    return tp, fp, fn


def _best_threshold(peaks: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 37)
    best_thr, best_f2 = 0.5, -1.0
    for thr in grid:
        tp, fp, fn = _counts_from_threshold(peaks, labels, float(thr))
        f2 = f2_score_from_counts(tp, fp, fn, beta=2.0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = float(thr)
    return best_thr, float(best_f2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate no-motor threshold")
    parser.add_argument("--config", type=str, default="./configs/config_laptop_8gb.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="./outputs_laptop/no_motor_calibration.json")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    ckpt = str(args.checkpoint or cfg["paths"]["finetune_ckpt"])
    device = resolve_device()

    val_df = pd.read_csv(cfg["paths"]["val_csv"])
    predictor = HeatmapPredictorFactory(cfg=cfg, ckpt_path=ckpt, device=device)

    peaks: list[float] = []
    labels: list[int] = []
    patch_size = tuple(cfg["data"]["patch_size"])

    for row in val_df.itertuples(index=False):
        volume = load_tomogram(str(row.tomo_path), cfg["data"]["normalization"])
        z, y, x = volume.shape
        pz, py, px = patch_size
        sz = max(0, (z - pz) // 2)
        sy = max(0, (y - py) // 2)
        sx = max(0, (x - px) // 2)
        crop = volume[sz:sz+pz, sy:sy+py, sx:sx+px]
        if crop.shape != patch_size:
            pad = [(0, pz - crop.shape[0]), (0, py - crop.shape[1]), (0, px - crop.shape[2])]
            crop = np.pad(crop, pad, mode="reflect")
        x_t = torch.from_numpy(crop[None, None, ...]).float().to(device)
        with torch.no_grad():
            heat = predictor.predict_heatmap(x_t)
            peaks.append(float(heat.max().item()))
        labels.append(int(row.has_motor))

    peak_arr = np.asarray(peaks, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    best_thr, best_f2 = _best_threshold(peak_arr, label_arr)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(peak_arr, label_arr)
    calibrated = iso.predict(peak_arr)
    cal_thr, cal_f2 = _best_threshold(np.asarray(calibrated), label_arr)

    payload = {
        "samples": int(len(label_arr)),
        "raw_best_threshold": best_thr,
        "raw_best_f2": best_f2,
        "isotonic_best_threshold": cal_thr,
        "isotonic_best_f2": cal_f2,
        "recommendation": {
            "no_motor_threshold": best_thr,
            "note": "Set inference.no_motor_threshold to raw_best_threshold unless isotonic mode is enabled.",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved calibration report to {out_path}")


if __name__ == "__main__":
    main()
