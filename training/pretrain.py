"""Stage-1 MAE self-supervised pretraining."""

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
from models.mae import MaskedAutoencoder3D


def run_pretraining(cfg: Dict[str, Any], device: torch.device) -> str:
    """Run MAE pretraining and save best checkpoint."""
    ds = TomogramDataset(
        csv_path=cfg["paths"]["train_csv"],
        patch_size=cfg["data"]["patch_size"],
        hard_negative_ratio=cfg["data"]["hard_negative_ratio"],
        boundary_margin=cfg["data"]["boundary_margin"],
        positive_radius=cfg["data"]["positive_radius"],
        augment_cfg=cfg["data"]["augment"],
        normalization_cfg=cfg["data"]["normalization"],
        is_train=True,
        seed=cfg["seed"],
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["stage1"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=True,
    )

    model = MaskedAutoencoder3D(cfg["model"]).to(device)
    opt = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["stage1"]["lr"]),
        weight_decay=float(cfg["training"]["stage1"]["weight_decay"]),
    )
    sched = CosineAnnealingLR(
        opt,
        T_max=int(cfg["training"]["stage1"]["epochs"]),
        eta_min=float(cfg["training"]["scheduler"]["eta_min"]),
    )
    scaler = GradScaler(enabled=bool(cfg["training"]["amp"]) and device.type == "cuda")
    out_path = Path(cfg["paths"]["pretrain_ckpt"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(int(cfg["training"]["stage1"]["epochs"])):
        model.train()
        running = 0.0
        for batch in tqdm(loader, desc=f"Pretrain {epoch + 1}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
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
            torch.save({"model": model.state_dict(), "epoch": epoch, "loss": best_loss}, out_path)
    return str(out_path)
