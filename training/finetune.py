"""Stage-2 multitask fine-tuning with curriculum and OHEM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TomogramDataset
from evaluation.metrics import evaluate_batch_localization, f2_score_from_counts
from models.multitask_model import build_multitask_model
from training.hard_negative import OnlineHardExampleMiner, curriculum_hard_negative_ratio
from training.losses import CombinedMotorLoss


def _build_loaders(cfg: Dict[str, Any], hard_negative_ratio: float | None = None) -> Tuple[DataLoader, DataLoader]:
    train_ds = TomogramDataset(
        csv_path=cfg["paths"]["train_csv"],
        patch_size=cfg["data"]["patch_size"],
        hard_negative_ratio=float(hard_negative_ratio if hard_negative_ratio is not None else cfg["data"]["hard_negative_ratio"]),
        boundary_margin=cfg["data"]["boundary_margin"],
        positive_radius=cfg["data"]["positive_radius"],
        augment_cfg=cfg["data"]["augment"],
        normalization_cfg=cfg["data"]["normalization"],
        is_train=True,
        seed=cfg["seed"],
        cache_mode=str(cfg.get("data", {}).get("cache", {}).get("mode", "memory")),
        max_cache_items=int(cfg.get("data", {}).get("cache", {}).get("max_items", 24)),
        validate_manifest_on_load=bool(cfg.get("data", {}).get("validate_manifest_on_load", True)),
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
        cache_mode=str(cfg.get("data", {}).get("cache", {}).get("mode", "memory")),
        max_cache_items=int(cfg.get("data", {}).get("cache", {}).get("max_items", 24)),
        validate_manifest_on_load=bool(cfg.get("data", {}).get("validate_manifest_on_load", True)),
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
    no_motor_thr = float(cfg.get("inference", {}).get("no_motor_threshold", 0.5))

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
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
            pred = 0.5 * pred_centroid + 0.5 * argmax_coords

            # Apply no-motor gating using heatmap peak.
            peaks = flat.max(dim=1).values
            pred[peaks < no_motor_thr] = 0.0

            batch_tp, batch_fp, batch_fn = evaluate_batch_localization(pred, coord_t, labels, tol)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
    return f2_score_from_counts(tp, fp, fn)


def run_finetuning(cfg: Dict[str, Any], device: torch.device, pretrained_ckpt: str | None = None) -> str:
    """Fine-tune the multitask model and return best checkpoint path."""
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
    scaler = amp.GradScaler(device="cuda", enabled=bool(cfg["training"]["amp"]) and device.type == "cuda")

    ohem_cfg = cfg.get("training", {}).get("ohem", {})
    miner = OnlineHardExampleMiner(
        base_weight=float(ohem_cfg.get("base_weight", 1.0)),
        hard_weight=float(ohem_cfg.get("hard_weight", 2.0)),
    )
    fp_thr = float(ohem_cfg.get("fp_threshold", 0.5))
    fp_penalty_weight = float(ohem_cfg.get("fp_penalty_weight", 0.1))

    curriculum_cfg = cfg.get("training", {}).get("curriculum", {})
    use_curriculum = bool(curriculum_cfg.get("enabled", True))
    min_ratio = float(curriculum_cfg.get("hard_negative_min_ratio", 0.1))
    max_ratio = float(curriculum_cfg.get("hard_negative_max_ratio", cfg["data"].get("hard_negative_ratio", 0.35)))

    out_path = Path(cfg["paths"]["finetune_ckpt"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_f2 = -1.0
    epochs = int(cfg["training"]["stage2"]["epochs"])

    val_loader_ref = None
    for epoch in range(epochs):
        if use_curriculum:
            hn_ratio = curriculum_hard_negative_ratio(epoch, epochs, min_ratio=min_ratio, max_ratio=max_ratio)
        else:
            hn_ratio = float(cfg["data"]["hard_negative_ratio"])

        train_loader, val_loader = _build_loaders(cfg, hard_negative_ratio=hn_ratio)
        val_loader_ref = val_loader

        model.train()
        loop = tqdm(train_loader, desc=f"Finetune {epoch + 1}", leave=False)
        for batch in loop:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["seg_target"].to(device, non_blocking=True)
            coord_t = batch["centroid"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                out = model(x)
                loss, logs = criterion(out["segmentation"], y, out["centroid"], coord_t, labels)

                seg_prob = torch.sigmoid(out["segmentation"]).detach()
                seg_peak = seg_prob[:, 0].flatten(1).max(dim=1).values
                sample_weights = miner.compute_weights(seg_peak=seg_peak, labels=labels.view(-1), threshold=fp_thr).to(device)
                fp_penalty = (torch.relu(seg_peak - fp_thr) * (labels.view(-1) < 0.5).float() * sample_weights).mean()
                loss = loss + fp_penalty_weight * fp_penalty

            scaler.scale(loss).backward()
            if cfg["training"].get("grad_clip_norm") is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
            scaler.step(opt)
            scaler.update()
            loop.set_postfix(
                total=float(loss.detach().item()),
                seg=float(logs["seg_loss"]),
                reg=float(logs["reg_loss"]),
                hn=f"{hn_ratio:.2f}",
                fp=f"{miner.recent_fp_rate():.3f}",
            )

        sched.step()
        val_f2 = validate(model, val_loader, cfg, device)
        if val_f2 > best_f2:
            best_f2 = val_f2
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f2": val_f2}, out_path)

    if val_loader_ref is None:
        raise RuntimeError("No validation loader constructed during finetuning.")
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
