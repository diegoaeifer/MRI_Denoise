"""
MONAI Bundle-based training entry point for MRI denoising.

Usage:
    python -m mri_denoise.train --config configs/train.yaml [--model nafnet]
    python -m mri_denoise.train --config configs/finetune_swinunetr.yaml
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim as optim
import yaml
from monai.data import CacheDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.datalist import build_datalist
from .data.transforms import build_train_transforms, build_val_transforms
from .engines import run_training
from .losses.composite import CompositeLoss
from .networks.registry import get_network
from .networks.swinunetr_denoising import get_layerwise_lr_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(network: torch.nn.Module, cfg: Dict[str, Any]) -> optim.Optimizer:
    training_cfg = cfg["training"]
    base_lr = float(training_cfg["lr"])
    opt_name = training_cfg.get("optimizer", "Adam")
    wd = float(training_cfg.get("weight_decay", 1e-5))

    # Use layerwise LR decay if the model supports set_epoch (e.g. SwinUNETRDenoising)
    if hasattr(network, "set_epoch") and training_cfg.get("layerwise_lr_decay"):
        decay = float(training_cfg["layerwise_lr_decay"])
        param_groups = get_layerwise_lr_groups(network, base_lr, decay)
    else:
        param_groups = network.parameters()  # type: ignore[assignment]

    if opt_name == "AdamW":
        return optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)
    return optim.Adam(param_groups, lr=base_lr, weight_decay=wd)


def build_scheduler(
    optimizer: optim.Optimizer, cfg: Dict[str, Any]
) -> optim.lr_scheduler._LRScheduler | None:
    training_cfg = cfg["training"]
    name = training_cfg.get("scheduler", "CosineAnnealingLR")
    if name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_cfg.get("scheduler_T_max", training_cfg["epochs"])
        )
    return None


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Data
    datalist = build_datalist(cfg["data"])
    train_ds = CacheDataset(
        data=datalist["train"],
        transform=build_train_transforms(cfg),
        cache_rate=cfg["data"].get("cache_rate", 1.0),
    )
    val_ds = CacheDataset(
        data=datalist["val"],
        transform=build_val_transforms(cfg),
        cache_rate=cfg["data"].get("cache_rate", 1.0),
    )
    batch = cfg["training"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw)

    # Network
    net_cfg = cfg["network"]
    network = get_network(
        net_cfg["name"],
        spatial_dims=cfg["spatial_dims"],
        in_channels=net_cfg["in_channels"],
        out_channels=net_cfg["out_channels"],
        **{k: v for k, v in net_cfg.items() if k not in ("name", "_target_")},
    ).to(device)

    # Loss
    loss_fn = CompositeLoss(
        spatial_dims=cfg["spatial_dims"],
        weights=cfg["losses"]["weights"],
        data_range=cfg["losses"].get("data_range", 1.0),
    ).to(device)

    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    run_id = f"{net_cfg['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(cfg["training"].get("output_dir", "experiments"), "logs", run_id)
    writer = SummaryWriter(log_dir=log_dir)

    run_training(network, loss_fn, optimizer, train_loader, val_loader,
                 device, cfg, scheduler, writer, run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Denoising — MONAI training")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    main(parser.parse_args())
