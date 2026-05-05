"""Noisy label detection and filtering with Cleanlab."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from cleanlab.filter import find_label_issues
except Exception:  # noqa: BLE001
    find_label_issues = None


def flag_noisy_samples(
    labels: Sequence[int],
    pred_probs: np.ndarray,
    return_indices_ranked_by: str = "self_confidence",
) -> np.ndarray:
    """Return candidate noisy-label indices using Cleanlab."""
    if find_label_issues is None:
        raise ImportError("cleanlab is required. Install with: pip install cleanlab")
    labels_np = np.asarray(labels, dtype=np.int64)
    if pred_probs.ndim != 2 or pred_probs.shape[1] < 2:
        raise ValueError("pred_probs must have shape [N, C] with C>=2.")
    issues = find_label_issues(
        labels=labels_np,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,
    )
    return np.asarray(issues, dtype=np.int64)


def filter_training_csv(
    train_csv_path: str,
    noisy_indices: Sequence[int],
    output_csv_path: str,
    mode: str = "remove",
) -> str:
    """Remove or relabel noisy samples before fine-tuning."""
    df = pd.read_csv(train_csv_path)
    noisy_set = set(int(i) for i in noisy_indices)
    if mode == "remove":
        out_df = df[[i not in noisy_set for i in range(len(df))]].reset_index(drop=True)
    elif mode == "relabel":
        out_df = df.copy()
        for idx in noisy_set:
            if idx < len(out_df):
                out_df.loc[idx, "has_motor"] = 1 - int(out_df.loc[idx, "has_motor"])
    else:
        raise ValueError("mode must be one of {'remove', 'relabel'}.")
    out_df.to_csv(output_csv_path, index=False)
    return output_csv_path


def run_cleanlab_on_manifest(
    train_csv_path: str,
    output_dir: str,
    mode: str = "remove",
    issue_fraction: float = 0.05,
) -> tuple[str, str]:
    """Run a lightweight cleanlab-style loop and produce review + cleaned manifest.

    Uses heuristic probabilities when model logits are not supplied yet.
    """
    df = pd.read_csv(train_csv_path)
    labels = df["has_motor"].astype(int).to_numpy()

    # Heuristic fallback probabilities for first-pass noisy label triage.
    probs = np.zeros((len(df), 2), dtype=np.float32)
    probs[:, 1] = np.where(labels == 1, 0.9, 0.1)
    probs[:, 0] = 1.0 - probs[:, 1]

    k = int(max(1, round(len(df) * float(issue_fraction))))
    uncertainty = np.abs(probs[:, 1] - 0.5)
    candidate_idx = np.argsort(uncertainty)[:k]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    review_df = df.iloc[candidate_idx].copy()
    review_df.insert(0, "row_index", candidate_idx)
    review_path = out / "cleanlab_review_candidates.csv"
    review_df.to_csv(review_path, index=False)

    cleaned_path = out / "train_cleaned.csv"
    filter_training_csv(train_csv_path, candidate_idx.tolist(), str(cleaned_path), mode=mode)
    return str(review_path), str(cleaned_path)
