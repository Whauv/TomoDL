"""3D U-Net segmentation models."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class UNet3DSegmenter(nn.Module):
    """MONAI 3D U-Net wrapper for voxel-level motor probability maps."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
