"""Dual-head multitask model for segmentation + centroid regression."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.encoder import build_resnet3d_encoder
from models.regression_head import CentroidRegressionHead


class ConvBlock(nn.Module):
    """Convolutional refinement block for decoder stages."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    """Upsample + skip-fusion stage."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiTaskMotorModel(nn.Module):
    """Shared encoder with a U-Net-style segmentation decoder and regression head."""

    def __init__(self, model_cfg: Dict, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.encoder = build_resnet3d_encoder(model_cfg)
        oc = self.encoder.out_channels
        self.dec3 = DecoderStage(oc[4], oc[3], oc[3])
        self.dec2 = DecoderStage(oc[3], oc[2], oc[2])
        self.dec1 = DecoderStage(oc[2], oc[1], oc[1])
        self.dec0 = DecoderStage(oc[1], oc[0], oc[0])
        self.seg_head = nn.Conv3d(oc[0], 1, kernel_size=1)
        self.reg_head = CentroidRegressionHead(
            in_channels=oc[4], hidden_dim=int(model_cfg.get("regression_hidden_dim", 128))
        )

    def load_pretrained_encoder(self, ckpt_path: str) -> None:
        """Load MAE-pretrained encoder weights into the shared encoder."""
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model", state)
        encoder_weights = {
            k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_weights, strict=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder.forward_features(x)
        x0, x1, x2, x3, x4 = feats
        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)
        seg = self.seg_head(d0)
        if seg.shape[2:] != x.shape[2:]:
            seg = nn.functional.interpolate(seg, size=x.shape[2:], mode="trilinear", align_corners=False)
        centroid = self.reg_head(x4)
        return {"segmentation": seg, "centroid": centroid}


def build_multitask_model(
    model_cfg: Dict,
    patch_size: tuple[int, int, int],
    pretrained_ckpt: Optional[str] = None,
) -> MultiTaskMotorModel:
    """Factory for multitask model with optional pretrained encoder loading."""
    model = MultiTaskMotorModel(model_cfg=model_cfg, patch_size=patch_size)
    if pretrained_ckpt:
        model.load_pretrained_encoder(pretrained_ckpt)
    return model
