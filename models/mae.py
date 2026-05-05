"""3D self-supervised models: MAE and contrastive pretraining."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import ResNet3DEncoder, build_resnet3d_encoder


class MAEDecoder3D(nn.Module):
    """Lightweight 3D decoder that reconstructs input volumes from latent features."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 1) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, hidden_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(hidden_channels),
            nn.GELU(),
            nn.ConvTranspose3d(hidden_channels, hidden_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(hidden_channels // 2),
            nn.GELU(),
            nn.ConvTranspose3d(hidden_channels // 2, hidden_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(hidden_channels // 4),
            nn.GELU(),
            nn.ConvTranspose3d(hidden_channels // 4, hidden_channels // 8, kernel_size=2, stride=2),
            nn.BatchNorm3d(hidden_channels // 8),
            nn.GELU(),
            nn.Conv3d(hidden_channels // 8, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class MaskedAutoencoder3D(nn.Module):
    """Masked autoencoder that masks latent 3D tokens and reconstructs missing voxels."""

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        mae_cfg = model_cfg.get("mae", {})
        self.mask_ratio = float(mae_cfg.get("mask_ratio", 0.75))
        self.encoder: ResNet3DEncoder = build_resnet3d_encoder(model_cfg)
        self.decoder = MAEDecoder3D(
            in_channels=self.encoder.out_channels[-1],
            hidden_channels=int(mae_cfg.get("decoder_channels", 128)),
            out_channels=int(model_cfg.get("encoder_in_channels", 1)),
        )

    def _mask_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, d, h, w = latent.shape
        token_count = d * h * w
        keep_count = max(1, int(token_count * (1.0 - self.mask_ratio)))
        noise = torch.rand((b, token_count), device=latent.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :keep_count]

        mask_flat = torch.ones((b, token_count), device=latent.device, dtype=latent.dtype)
        mask_flat.scatter_(1, ids_keep, 0.0)
        mask = mask_flat.view(b, 1, d, h, w)
        masked_latent = latent * (1.0 - mask)
        return masked_latent, mask

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encoder(x)
        masked_latent, latent_mask = self._mask_latent(latent)
        recon = self.decoder(masked_latent)
        recon = F.interpolate(recon, size=x.shape[2:], mode="trilinear", align_corners=False)
        voxel_mask = F.interpolate(latent_mask, size=x.shape[2:], mode="nearest")
        masked_voxels = voxel_mask > 0.5
        loss = F.mse_loss(recon[masked_voxels], x[masked_voxels]) if masked_voxels.any() else F.mse_loss(recon, x)
        return {"loss": loss, "reconstruction": recon, "mask": voxel_mask}


class ContrastivePretrainer3D(nn.Module):
    """SimCLR-style 3D contrastive pretrainer over shared ResNet3D encoder."""

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        ssl_cfg = model_cfg.get("ssl", {})
        self.temperature = float(ssl_cfg.get("temperature", 0.2))
        proj_dim = int(ssl_cfg.get("projection_dim", 128))
        self.encoder: ResNet3DEncoder = build_resnet3d_encoder(model_cfg)
        feat_dim = int(self.encoder.out_channels[-1])
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        pooled = F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
        z = self.projector(pooled)
        return F.normalize(z, dim=1)

    def nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        b = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / max(self.temperature, 1e-6)
        eye = torch.eye(2 * b, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)

        target = torch.arange(2 * b, device=sim.device)
        target = (target + b) % (2 * b)
        loss = F.cross_entropy(sim, target)
        return loss

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> Dict[str, torch.Tensor]:
        z1 = self.encode(view1)
        z2 = self.encode(view2)
        loss = self.nt_xent(z1, z2)
        return {"loss": loss, "z1": z1, "z2": z2}
