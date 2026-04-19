"""DETR-style 3D detection model for ensemble predictions."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PositionalEncoding3D(nn.Module):
    """Simple learnable positional embedding for 3D tokens."""

    def __init__(self, channels: int, max_tokens: int = 8192) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_tokens, channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[1]
        idx = torch.arange(n, device=tokens.device)
        return tokens + self.embedding(idx)[None, :, :]


class DETR3D(nn.Module):
    """Compact DETR-3D detector producing heatmap-like outputs."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, num_layers: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.pos = PositionalEncoding3D(channels=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.proj(x)
        b, c, d, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.pos(tokens)
        out = self.transformer(tokens)
        heat = self.head(out).transpose(1, 2).reshape(b, 1, d, h, w)
        heat = nn.functional.interpolate(heat, size=x.shape[2:], mode="trilinear", align_corners=False)
        return {"heatmap": heat}
