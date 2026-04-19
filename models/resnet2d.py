"""2D slice-based ResNet-50 used for ensemble inference."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet2DSliceModel(nn.Module):
    """Apply ResNet-50 on tomogram slices and project back to 3D heatmaps."""

    def __init__(self, in_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Conv2d(2048, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, d, h, w = x.shape
        xs = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        f = self.features(xs)
        logits = self.head(f)
        logits = nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        logits = logits.view(b, d, 1, h, w).permute(0, 2, 1, 3, 4)
        return {"heatmap": logits}
