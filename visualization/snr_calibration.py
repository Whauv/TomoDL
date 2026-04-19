"""SNR calibration and failure-case visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def compute_snr(volume: np.ndarray) -> float:
    """Estimate SNR as mean absolute signal over noise standard deviation."""
    signal = float(np.mean(np.abs(volume)))
    noise = float(np.std(volume)) + 1e-6
    return signal / noise


def plot_snr_calibration(
    snr_values: Sequence[float],
    f2_values: Sequence[float],
    output_path: str,
    bins: int = 8,
) -> None:
    """Plot binned SNR vs detection F2 calibration curve."""
    snr = np.asarray(snr_values, dtype=np.float32)
    f2 = np.asarray(f2_values, dtype=np.float32)
    edges = np.linspace(float(snr.min()), float(snr.max()) + 1e-6, bins + 1)
    bin_ids = np.digitize(snr, edges) - 1
    xs, ys = [], []
    for i in range(bins):
        mask = bin_ids == i
        if np.any(mask):
            xs.append(float(snr[mask].mean()))
            ys.append(float(f2[mask].mean()))
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("SNR (binned mean)")
    plt.ylabel("F2")
    plt.title("SNR Calibration Curve")
    plt.grid(alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def visualize_failure_cases(
    volumes: Sequence[np.ndarray],
    pred_coords: Sequence[Sequence[float]],
    true_coords: Sequence[Sequence[float]],
    output_path: str,
    max_cases: int = 6,
) -> None:
    """Render central 2D slices for false positives/negatives with coordinate overlays."""
    n = min(max_cases, len(volumes))
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(n):
        vol = volumes[i]
        z_center = int(np.clip(round(true_coords[i][0]), 0, vol.shape[0] - 1))
        sl = vol[z_center]
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(sl, cmap="gray")
        py, px = pred_coords[i][1], pred_coords[i][2]
        ty, tx = true_coords[i][1], true_coords[i][2]
        ax.scatter([px], [py], c="r", s=35, label="pred")
        ax.scatter([tx], [ty], c="lime", s=35, label="true")
        ax.set_title(f"Case {i + 1} z={z_center}")
        ax.axis("off")
        ax.legend(loc="lower right", fontsize=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
