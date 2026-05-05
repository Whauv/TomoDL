"""Stage-1 self-supervised pretraining (MAE or contrastive)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TomogramDataset
from models.mae import ContrastivePretrainer3D, MaskedAutoencoder3D


def _build_ssl_augment_cfg(base_cfg: Dict[str, Any], strong: bool) -> Dict[str, Any]:
    aug = dict(base_cfg)
    if not strong:
        return aug
    aug["enabled"] = True
    aug["elastic_prob"] = max(float(aug.get("elastic_prob", 0.2)), 0.35)
    aug["flip_prob"] = max(float(aug.get("flip_prob", 0.5)), 0.6)
    aug["noise_prob"] = max(float(aug.get("noise_prob", 0.2)), 0.35)
    aug["contrast_prob"] = max(float(aug.get("contrast_prob", 0.2)), 0.35)
    return aug


def run_pretraining(cfg: Dict[str, Any], device: torch.device) -> str:
    """Run SSL pretraining and save best checkpoint."""
    stage1_cfg = cfg["training"]["stage1"]
    ssl_mode = str(stage1_cfg.get("ssl_mode", "mae")).lower()
    strong_aug = bool(stage1_cfg.get("strong_ssl_augment", True))

    ds = TomogramDataset(
        csv_path=cfg["paths"]["train_csv"],
        patch_size=cfg["data"]["patch_size"],
        hard_negative_ratio=cfg["data"]["hard_negative_ratio"],
        boundary_margin=cfg["data"]["boundary_margin"],
        positive_radius=cfg["data"]["positive_radius"],
        augment_cfg=_build_ssl_augment_cfg(cfg["data"]["augment"], strong_aug),
        normalization_cfg=cfg["data"]["normalization"],
        is_train=True,
        seed=cfg["seed"],
        cache_mode=str(cfg.get("data", {}).get("cache", {}).get("mode", "memory")),
        max_cache_items=int(cfg.get("data", {}).get("cache", {}).get("max_items", 24)),
        validate_manifest_on_load=bool(cfg.get("data", {}).get("validate_manifest_on_load", True)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(stage1_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=True,
    )

    if ssl_mode == "contrastive":
        model = ContrastivePretrainer3D(cfg["model"]).to(device)
    else:
        model = MaskedAutoencoder3D(cfg["model"]).to(device)

    opt = AdamW(
        model.parameters(),
        lr=float(stage1_cfg["lr"]),
        weight_decay=float(stage1_cfg["weight_decay"]),
    )
    sched = CosineAnnealingLR(
        opt,
        T_max=int(stage1_cfg["epochs"]),
        eta_min=float(cfg["training"]["scheduler"]["eta_min"]),
    )
    scaler = GradScaler(enabled=bool(cfg["training"]["amp"]) and device.type == "cuda")
    out_path = Path(cfg["paths"]["pretrain_ckpt"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(int(stage1_cfg["epochs"])):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"Pretrain {epoch + 1}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                if ssl_mode == "contrastive":
                    noise1 = torch.randn_like(x) * 0.05
                    noise2 = torch.randn_like(x) * 0.05
                    out = model(torch.clamp(x + noise1, -4.0, 4.0), torch.clamp(x + noise2, -4.0, 4.0))
                else:
                    out = model(x)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach().item())
        sched.step()
        epoch_loss = running / max(1, len(loader))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": best_loss, "ssl_mode": ssl_mode}, out_path)
    return str(out_path)
