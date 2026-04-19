"""Noisy label detection and filtering with Cleanlab."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues


def flag_noisy_samples(
    labels: Sequence[int],
    pred_probs: np.ndarray,
    return_indices_ranked_by: str = "self_confidence",
) -> np.ndarray:
    """Return candidate noisy-label indices using Cleanlab."""
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
