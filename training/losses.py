"""Loss functions for multitask motor localization."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from monai.losses import DiceCELoss


class CombinedMotorLoss(nn.Module):
    """Dice+BCE segmentation loss plus SmoothL1 centroid regression."""

    def __init__(self, w1_seg: float = 0.7, w2_reg: float = 0.3) -> None:
        super().__init__()
        self.w1_seg = float(w1_seg)
        self.w2_reg = float(w2_reg)
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=0.5, lambda_ce=0.5)
        self.reg_loss = nn.SmoothL1Loss(beta=1.0, reduction="none")

    def forward(
        self,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        coord_pred: torch.Tensor,
        coord_target: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seg = self.seg_loss(seg_pred, seg_target)
        per_sample = self.reg_loss(coord_pred, coord_target).mean(dim=1)
        sample_weights = labels.view(-1)
        reg = (per_sample * sample_weights).sum() / (sample_weights.sum() + 1e-6)
        total = self.w1_seg * seg + self.w2_reg * reg
        return total, {"seg_loss": seg.detach(), "reg_loss": reg.detach(), "total_loss": total.detach()}
