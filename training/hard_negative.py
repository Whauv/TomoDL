"""Hard-negative sampling helpers."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_hard_negative_sampler(labels: List[int], hard_negative_weight: float = 1.8) -> WeightedRandomSampler:
    """Construct a weighted sampler to oversample hard negatives."""
    labels_np = np.asarray(labels, dtype=np.int64)
    weights = np.ones_like(labels_np, dtype=np.float64)
    weights[labels_np == 0] = hard_negative_weight
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
