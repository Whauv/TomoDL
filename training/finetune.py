"""Stage-2 multitask fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TomogramDataset
from evaluation.metrics import evaluate_batch_localization, f2_score_from_counts
from models.multitask_model import build_multitask_model
from training.losses import CombinedMotorLoss


def _build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    train_ds = TomogramDataset(
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
    val_ds = TomogramDataset(
        csv_path=cfg["paths"]["val_csv"],
        patch_size=cfg["data"]["patch_size"],
        hard_negative_ratio=0.0,
        boundary_margin=cfg["data"]["boundary_margin"],
        positive_radius=cfg["data"]["positive_radius"],
        augment_cfg={"enabled": False},
        normalization_cfg=cfg["data"]["normalization"],
        is_train=False,
        seed=cfg["seed"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["stage2"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["inference"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )
    return train_loader, val_loader


def validate(model: torch.nn.Module, loader: DataLoader, cfg: Dict[str, Any], device: torch.device) -> float:
    """Compute validation F2 based on centroid localization."""
    model.eval()
    tp, fp, fn = 0, 0, 0
    tol = float(cfg["evaluation"]["voxel_tolerance"])
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["seg_target"].to(device, non_blocking=True)
            coord_t = batch["centroid"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).view(-1)

            out = model(x)
            seg_prob = torch.sigmoid(out["segmentation"])
            pred_centroid = out["centroid"]
            flat = seg_prob[:, 0].flatten(1)
            idx = flat.argmax(dim=1)
            d, h, w = seg_prob.shape[2:]
            z = idx // (h * w)
            y = (idx % (h * w)) // w
            x_coord = idx % w
            argmax_coords = torch.stack([z, y, x_coord], dim=1).float()
            # Blend direct regression with segmentation argmax for robust localization.
            pred = 0.5 * pred_centroid + 0.5 * argmax_coords
            batch_tp, batch_fp, batch_fn = evaluate_batch_localization(pred, coord_t, labels, tol)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
    return f2_score_from_counts(tp, fp, fn)


def run_finetuning(cfg: Dict[str, Any], device: torch.device, pretrained_ckpt: str | None = None) -> str:
    """Fine-tune the multitask model and return best checkpoint path."""
    train_loader, val_loader = _build_loaders(cfg)
    model = build_multitask_model(
        cfg["model"], patch_size=tuple(cfg["data"]["patch_size"]), pretrained_ckpt=pretrained_ckpt
    ).to(device)

    lw = cfg["training"]["stage2"]["loss_weights"]
    criterion = CombinedMotorLoss(w1_seg=float(lw["w1_seg"]), w2_reg=float(lw["w2_reg"]))
    opt = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["stage2"]["lr"]),
        weight_decay=float(cfg["training"]["stage2"]["weight_decay"]),
    )
    sched = CosineAnnealingLR(
        opt,
        T_max=int(cfg["training"]["scheduler"]["t_max"]),
        eta_min=float(cfg["training"]["scheduler"]["eta_min"]),
    )
    scaler = GradScaler(enabled=bool(cfg["training"]["amp"]) and device.type == "cuda")

    out_path = Path(cfg["paths"]["finetune_ckpt"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_f2 = -1.0

    for epoch in range(int(cfg["training"]["stage2"]["epochs"])):
        model.train()
        loop = tqdm(train_loader, desc=f"Finetune {epoch + 1}", leave=False)
        for batch in loop:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["seg_target"].to(device, non_blocking=True)
            coord_t = batch["centroid"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                out = model(x)
                loss, logs = criterion(out["segmentation"], y, out["centroid"], coord_t, labels)
            scaler.scale(loss).backward()
            if cfg["training"]["grad_clip_norm"] is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
            scaler.step(opt)
            scaler.update()
            loop.set_postfix(total=float(logs["total_loss"]), seg=float(logs["seg_loss"]), reg=float(logs["reg_loss"]))

        sched.step()
        val_f2 = validate(model, val_loader, cfg, device)
        if val_f2 > best_f2:
            best_f2 = val_f2
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f2": val_f2}, out_path)
    return str(out_path)


def train_and_validate_once(
    cfg: Dict[str, Any],
    device: torch.device,
    pretrained_ckpt: str | None = None,
    max_epochs: int | None = None,
) -> float:
    """Utility for hyperparameter tuning. Returns best validation F2."""
    if max_epochs is not None:
        cfg = {**cfg, "training": {**cfg["training"], "stage2": {**cfg["training"]["stage2"], "epochs": max_epochs}}}
    ckpt = run_finetuning(cfg, device, pretrained_ckpt=pretrained_ckpt)
    state = torch.load(ckpt, map_location="cpu")
    return float(state.get("val_f2", 0.0))
