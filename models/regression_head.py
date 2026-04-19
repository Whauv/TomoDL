"""Regression head for motor centroid prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class CentroidRegressionHead(nn.Module):
    """Predicts (z, y, x) centroid coordinates from encoder feature maps."""

    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(x))
