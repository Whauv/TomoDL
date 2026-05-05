"""Automated FP/FN slicing and interpretability visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from visualization.snr_calibration import compute_snr


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def slice_failures(metrics_csv: str, out_dir: str, tol: float = 10.0) -> str:
    df = pd.read_csv(metrics_csv)
    required = {"tomo_id", "pred_z", "pred_y", "pred_x", "true_z", "true_y", "true_x", "has_motor", "snr", "volume_size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics csv missing columns: {sorted(missing)}")

    pred = df[["pred_z", "pred_y", "pred_x"]].to_numpy(dtype=np.float32)
    true = df[["true_z", "true_y", "true_x"]].to_numpy(dtype=np.float32)
    dist = np.linalg.norm(pred - true, axis=1)
    df["dist"] = dist

    df["outcome"] = "tn"
    gt = df["has_motor"].astype(int).to_numpy()
    pred_has = (dist <= float(tol)).astype(int)
    df.loc[(gt == 1) & (pred_has == 1), "outcome"] = "tp"
    df.loc[(gt == 1) & (pred_has == 0), "outcome"] = "fn"
    df.loc[(gt == 0) & (pred_has == 1), "outcome"] = "fp"

    summary = (
        df.groupby(["outcome"]).size().rename("count").reset_index().sort_values("count", ascending=False)
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "failure_summary.csv"
    df_path = out / "failure_slices.csv"
    summary.to_csv(summary_path, index=False)
    df.to_csv(df_path, index=False)

    # By-SNR and volume complexity breakdown.
    df["snr_bin"] = pd.qcut(df["snr"], q=min(5, max(2, df["snr"].nunique())), duplicates="drop")
    df["size_bin"] = pd.qcut(df["volume_size"], q=min(5, max(2, df["volume_size"].nunique())), duplicates="drop")
    snr_break = df.groupby(["snr_bin", "outcome"]).size().unstack(fill_value=0)
    size_break = df.groupby(["size_bin", "outcome"]).size().unstack(fill_value=0)
    snr_break.to_csv(out / "failure_by_snr.csv")
    size_break.to_csv(out / "failure_by_size.csv")
    return str(df_path)


def simple_saliency(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    if "fused_heatmap" in out:
        score = out["fused_heatmap"].max()
    else:
        score = torch.sigmoid(out["segmentation"]).max()
    score.backward()
    grad = x.grad.detach().abs()[0, 0].cpu().numpy()
    grad = grad / (grad.max() + 1e-8)
    return grad


def save_saliency_slice(volume: np.ndarray, sal: np.ndarray, out_png: str) -> None:
    z = int(np.argmax(sal.mean(axis=(1, 2))))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(volume[z], cmap="gray")
    ax[0].set_title("Volume slice")
    ax[0].axis("off")
    ax[1].imshow(volume[z], cmap="gray")
    ax[1].imshow(sal[z], cmap="hot", alpha=0.5)
    ax[1].set_title("Saliency overlay")
    ax[1].axis("off")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure slicing and saliency helper")
    parser.add_argument("--metrics-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./outputs_laptop/failure_analytics")
    args = parser.parse_args()
    out = slice_failures(args.metrics_csv, args.out_dir)
    print(f"Saved failure analytics to: {out}")


if __name__ == "__main__":
    main()
