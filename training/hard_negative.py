"""Hard-negative mining helpers: sampler + online curriculum utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
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


def curriculum_hard_negative_ratio(epoch_idx: int, total_epochs: int, min_ratio: float, max_ratio: float) -> float:
    """Linear curriculum from easy negatives to hard negatives."""
    if total_epochs <= 1:
        return float(max_ratio)
    t = max(0.0, min(1.0, float(epoch_idx) / float(total_epochs - 1)))
    return float(min_ratio + (max_ratio - min_ratio) * t)


@dataclass
class OnlineHardExampleMiner:
    """Tracks recent false positives and returns per-sample reweighting factors."""

    base_weight: float = 1.0
    hard_weight: float = 2.0
    history: List[float] = field(default_factory=list)

    def compute_weights(self, seg_peak: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return per-sample weights; boost confident false positives.

        False positive here means no-motor label but high segmentation peak.
        """
        peak = seg_peak.detach().view(-1)
        lbl = labels.detach().view(-1)
        is_fp = (lbl < 0.5) & (peak >= float(threshold))
        weights = torch.full_like(peak, fill_value=float(self.base_weight), dtype=torch.float32)
        weights[is_fp] = float(self.hard_weight)
        if peak.numel() > 0:
            self.history.append(float(is_fp.float().mean().item()))
            self.history = self.history[-200:]
        return weights

    def recent_fp_rate(self) -> float:
        """Return rolling average of false-positive rate proxy."""
        if not self.history:
            return 0.0
        return float(np.mean(np.asarray(self.history, dtype=np.float32)))
