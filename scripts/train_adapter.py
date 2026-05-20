"""Fine-tune a pretrained MRI denoiser with LoRA or FouRA adapters.

Usage:
    python scripts/train_adapter.py --config configs/config_lora_mri_finetune.yaml
    python scripts/train_adapter.py --config configs/config_foura_mri_finetune.yaml --dry_run
"""
from __future__ import annotations
import argparse, sys, datetime, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.optim as optim
from src.models.nafnet import NAFNet
from src.models.drunet import DRUNet
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Inlined from src.mri_denoise.train to avoid module-level ignite import chain
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(network: torch.nn.Module, cfg: Dict[str, Any]) -> optim.Optimizer:
    training_cfg = cfg["training"]
    base_lr = float(training_cfg["lr"])
    opt_name = training_cfg.get("optimizer", "Adam")
    wd = float(training_cfg.get("weight_decay", 1e-5))
    if opt_name == "AdamW":
        return optim.AdamW(network.parameters(), lr=base_lr, weight_decay=wd)
    return optim.Adam(network.parameters(), lr=base_lr, weight_decay=wd)


def build_scheduler(optimizer: optim.Optimizer, cfg: Dict[str, Any]):
    training_cfg = cfg["training"]
    name = training_cfg.get("scheduler", "CosineAnnealingLR")
    if name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_cfg.get("scheduler_T_max", training_cfg["epochs"])
        )
    return None


def build_base_model(net_cfg: dict) -> torch.nn.Module:
    name = net_cfg["name"].lower().replace("-", "_")
    if "nafnet" in name:
        return NAFNet(
            img_channel=net_cfg.get("in_channels", 2),
            width=net_cfg.get("width", 64),
            enc_blk_nums=net_cfg.get("enc_blk_nums", [2, 2, 4, 8]),
            middle_blk_num=net_cfg.get("middle_blk_num", 12),
            dec_blk_nums=net_cfg.get("dec_blk_nums", [2, 2, 2, 2]),
        )
    if "drunet" in name:
        return DRUNet(in_channels=net_cfg.get("in_channels", 2), out_channels=1)
    raise ValueError(f"Unknown base model: {name}")


def attach_adapter(model: torch.nn.Module, adapter_cfg: dict) -> torch.nn.Module:
    adapter_type = adapter_cfg.get("type", "lora").lower()
    rank = adapter_cfg.get("rank", 16)
    alpha = float(adapter_cfg.get("alpha", 32.0))
    targets = adapter_cfg.get("target_modules") or None

    # Freeze backbone first
    for p in model.parameters():
        p.requires_grad = False

    if adapter_type == "lora":
        from src.models.lora_adapter import attach_lora
        model = attach_lora(model, rank=rank, alpha=alpha, target_modules=targets)
    elif adapter_type == "foura":
        from src.models.foura_adapter import attach_foura
        model = attach_foura(model, rank=rank, alpha=alpha, target_modules=targets)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    return model


def normalize_cfg(cfg: dict) -> dict:
    """Translate adapter config keys to match existing train.py expectations."""
    training = cfg.get("training", {})
    # train.py build_optimizer expects "lr" not "learning_rate"
    if "learning_rate" in training and "lr" not in training:
        training["lr"] = training["learning_rate"]
    # train.py build_scheduler expects "CosineAnnealingLR" not "CosineAnnealing"
    if training.get("scheduler") == "CosineAnnealing":
        training["scheduler"] = "CosineAnnealingLR"
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg = normalize_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_cfg = cfg["network"].copy()
    weights_path = net_cfg.pop("pretrained_weights", None)
    # If pretrained_in_channels differs from in_channels, the first conv is expanded
    # after weight loading (zero-init the new gmap/noise channel for continual adaptation).
    pretrained_in_ch = net_cfg.pop("pretrained_in_channels", None)
    target_in_ch = net_cfg.get("in_channels", 2)

    if pretrained_in_ch and pretrained_in_ch != target_in_ch:
        # Build with original pretrained channel count, load weights, then expand
        load_cfg = {**net_cfg, "in_channels": pretrained_in_ch}
        model = build_base_model(load_cfg)
        if weights_path and Path(weights_path).exists():
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
            print(f"Loaded pretrained weights from {weights_path} ({pretrained_in_ch}ch)")
        else:
            print(f"[WARN] No pretrained weights at {weights_path} — starting from scratch")
        from src.models.lora_adapter import expand_input_channels
        model = expand_input_channels(model, new_in_channels=target_in_ch)
        print(f"Expanded input channels {pretrained_in_ch} -> {target_in_ch} (noise/gmap ch zero-initialized)")
    else:
        model = build_base_model(net_cfg)
        if weights_path and Path(weights_path).exists():
            model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            print(f"[WARN] No pretrained weights at {weights_path} — starting from scratch")

    adapter_cfg = cfg.get("adapter", {})
    adapter_type = adapter_cfg.get("type", "lora")
    model = attach_adapter(model, adapter_cfg)
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Adapter={adapter_type}  rank={adapter_cfg.get('rank')}  alpha={adapter_cfg.get('alpha')}")
    print(f"Total parameters:     {total:>12,}")
    print(f"Trainable parameters: {trainable:>12,}  ({trainable/total:.2%})")

    if args.dry_run:
        print("[dry_run] Exiting without training.")
        return

    # --- Full training path (requires data/finetune_split.json) ---
    from monai.data import CacheDataset, DataLoader
    from src.mri_denoise.data.datalist import build_datalist
    from src.mri_denoise.data.transforms import build_train_transforms, build_val_transforms
    from src.mri_denoise.losses.composite import CompositeLoss
    from src.mri_denoise.engines import run_training  # ignite required
    from torch.utils.tensorboard import SummaryWriter

    datalist = build_datalist(cfg["data"])
    train_ds = CacheDataset(data=datalist["train"], transform=build_train_transforms(cfg),
                            cache_rate=cfg["data"].get("cache_rate", 0.1))
    val_ds = CacheDataset(data=datalist["val"], transform=build_val_transforms(cfg),
                          cache_rate=cfg["data"].get("cache_rate", 0.1))
    batch = cfg["training"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw)

    loss_fn = CompositeLoss(
        spatial_dims=2,
        weights=cfg["losses"]["weights"],
        data_range=cfg["losses"].get("data_range", 1.0),
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    run_id = f"{cfg.get('run_id', 'adapter')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(cfg.get("output", {}).get("log_dir", "runs"), run_id)
    writer = SummaryWriter(log_dir=log_dir)

    run_training(model, loss_fn, optimizer, train_loader, val_loader,
                 device, cfg, scheduler, writer, run_id)


if __name__ == "__main__":
    main()
