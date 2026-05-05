"""DETR-style 3D detection model with query branch and uncertainty head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding3D(nn.Module):
    """Simple learnable positional embedding for 3D tokens."""

    def __init__(self, channels: int, max_tokens: int = 8192) -> None:
        super().__init__()
        self.channels = channels
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(max_tokens, channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[1]
        if n <= self.max_tokens:
            idx = torch.arange(n, device=tokens.device)
            return tokens + self.embedding(idx)[None, :, :]

        pos_idx = torch.arange(n, device=tokens.device, dtype=torch.float32).unsqueeze(1)
        half_dim = max(1, self.channels // 2)
        div_term = torch.exp(
            torch.arange(half_dim, device=tokens.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=tokens.device)) / max(1, half_dim - 1))
        )
        pe = torch.zeros((n, self.channels), device=tokens.device, dtype=tokens.dtype)
        pe[:, 0 : 2 * half_dim : 2] = torch.sin(pos_idx * div_term).to(tokens.dtype)
        pe[:, 1 : 2 * half_dim : 2] = torch.cos(pos_idx * div_term).to(tokens.dtype)
        return tokens + pe.unsqueeze(0)


class DETR3D(nn.Module):
    """Compact DETR-3D detector with dense and query outputs."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, num_layers: int = 4, num_queries: int = 32) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.pos = PositionalEncoding3D(channels=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.heat_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # z,y,x,objectness(logit)
        )
        self.query_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.proj(x)
        b, c, d, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.pos(tokens)
        memory = self.transformer(tokens)

        heat = self.heat_head(memory).transpose(1, 2).reshape(b, 1, d, h, w)
        heat = F.interpolate(heat, size=x.shape[2:], mode="trilinear", align_corners=False)

        queries = self.query_embed.weight[None, :, :].expand(b, -1, -1)
        pooled = memory.mean(dim=1, keepdim=True)
        qfeat = queries + pooled
        qraw = self.query_mlp(qfeat)
        coords = torch.sigmoid(qraw[..., :3])
        objectness = qraw[..., 3:4]
        uncertainty = F.softplus(self.query_uncertainty(qfeat))

        scale = torch.tensor([x.shape[2], x.shape[3], x.shape[4]], device=x.device, dtype=x.dtype)[None, None, :]
        coords_voxel = coords * scale

        return {
            "heatmap": heat,
            "query_coords": coords_voxel,
            "query_objectness": objectness,
            "query_uncertainty": uncertainty,
        }
