"""3D ResNet encoders used across TomoDL models."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    """Basic 3D residual block."""

    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNet3DEncoder(nn.Module):
    """ResNet-18/34 style 3D encoder returning hierarchical features."""

    def __init__(
        self,
        depth: str = "resnet18",
        in_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if depth == "resnet18":
            layers = [2, 2, 2, 2]
        elif depth == "resnet34":
            layers = [3, 4, 6, 3]
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, layers[3], stride=2)
        self.out_channels = [base_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [BasicBlock3D(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)
        x = self.pool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[-1]


def build_resnet3d_encoder(cfg: dict) -> ResNet3DEncoder:
    """Factory for a config-driven 3D ResNet encoder."""
    return ResNet3DEncoder(
        depth=str(cfg.get("encoder_name", "resnet18")),
        in_channels=int(cfg.get("encoder_in_channels", 1)),
        base_channels=int(cfg.get("encoder_base_channels", 32)),
    )
