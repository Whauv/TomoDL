"""Hybrid detector that fuses dense segmentation and query-based detection."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.detr3d import DETR3D
from models.multitask_model import build_multitask_model


class HybridMotorDetector(nn.Module):
    """Fuse U-Net dense heatmap and DETR query heatmap for robust localization."""

    def __init__(self, model_cfg: Dict, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.dense = build_multitask_model(model_cfg, patch_size=patch_size)
        self.query = DETR3D(
            in_channels=int(model_cfg.get("encoder_in_channels", 1)),
            hidden_dim=int(model_cfg.get("hybrid", {}).get("query_hidden_dim", 128)),
            num_layers=int(model_cfg.get("hybrid", {}).get("query_layers", 4)),
            num_queries=int(model_cfg.get("hybrid", {}).get("num_queries", 32)),
        )
        self.fusion_alpha = float(model_cfg.get("hybrid", {}).get("fusion_alpha", 0.7))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        dense_out = self.dense(x)
        query_out = self.query(x)
        dense_heat = torch.sigmoid(dense_out["segmentation"])
        query_heat = torch.sigmoid(query_out["heatmap"])
        alpha = max(0.0, min(1.0, self.fusion_alpha))
        fused_heat = alpha * dense_heat + (1.0 - alpha) * query_heat

        return {
            "segmentation": dense_out["segmentation"],
            "centroid": dense_out["centroid"],
            "query_coords": query_out["query_coords"],
            "query_objectness": query_out["query_objectness"],
            "query_uncertainty": query_out["query_uncertainty"],
            "fused_heatmap": fused_heat,
        }
