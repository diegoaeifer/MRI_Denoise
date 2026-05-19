"""
FouRA fine-tuning for pretrained denoisers on IXI dataset.

Trains pretrained models using Fourier Low-Rank Adaptation with:
- Loss: L1 + SSIM + DreamSim
- Input: magnitude + g-map (noise map)
- Dataset: IXI (10 volumes per sequence type)
- 2D/3D: Respects model architecture (2D models as 2D, 3D as 3D)

Usage:
    python -m mri_denoise.train_fouRA \
        --model drunet \
        --ixi-root /path/to/ixi \
        --sequence T1 \
        --spatial-dims 2 \
        --rank 16 \
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from monai.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_fouRA_model(
    model_name: str,
    spatial_dims: int = 2,
    rank: int = 16,
    alpha: float = 1.0,
) -> nn.Module:
    """
    Create a FouRA-adapted pretrained denoiser.

    Args:
        model_name: Model from registry (drunet, nafnet, swinunetr, etc.)
        spatial_dims: 2 or 3
        rank: FouRA rank
        alpha: FouRA scaling

    Returns:
        FouRA-adapted model
    """
    from .networks.registry import get_network
    from .adapters import create_fouRA_model

    # Load pretrained model
    model = get_network(
        model_name,
        spatial_dims=spatial_dims,
        in_channels=2,  # magnitude + gmap
        out_channels=1,  # denoised magnitude
    )
    logger.info(f"✓ Loaded {model_name} (spatial_dims={spatial_dims})")

    # Wrap with FouRA
    fouRA_model = create_fouRA_model(
        model=model,
        rank=rank,
        alpha=alpha,
        freeze_base=True,
    )
    fouRA_model.print_trainable_params()

    return fouRA_model


def build_loss_fn(spatial_dims: int = 2, device: str = "cuda") -> nn.Module:
    """
    Build composite loss: L1 + SSIM + DreamSim (equal weights).

    Args:
        spatial_dims: 2 or 3 for SSIM
        device: Device for loss computation

    Returns:
        Composite loss module
    """
    from .losses.composite import CompositeLoss
    from .metrics.dreamsim_metric import DreamSimMetric

    # L1 + SSIM with equal weights
    composite_loss = CompositeLoss(
        spatial_dims=spatial_dims,
        weights={
            "l1": 1.0,
            "ssim": 1.0,  # Equal to L1
            "psnr": 0.0,  # Disabled
            "charbonnier": 0.0,
            "perceptual": 0.0,
        },
        data_range=1.0,
    )

    # Add DreamSim weight (equal to L1 and SSIM)
    dreamsim = DreamSimMetric(device=device).eval()

    logger.info("✓ Loss: L1 (1.0) + SSIM (1.0) + DreamSim (1.0)")

    return composite_loss, dreamsim


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    dreamsim_fn: Optional[nn.Module],
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: str = "cuda",
    amp_enabled: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: FouRA-adapted model
        train_loader: Training data loader
        loss_fn: Composite loss (L1 + SSIM)
        dreamsim_fn: DreamSim metric (optional)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        amp_enabled: Use automatic mixed precision

    Returns:
        Dict with loss values
    """
    model.train()
    total_loss = 0.0
    l1_loss_total = 0.0
    ssim_loss_total = 0.0
    dreamsim_loss_total = 0.0
    num_batches = 0

    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    for batch_idx, batch in enumerate(train_loader):
        if isinstance(batch, dict):
            # Magnitude + gmap as input
            noisy = batch["image"].to(device)  # (B, 2, H, W) or (B, 2, H, W, D)
            clean = batch["label"].to(device)  # (B, 1, H, W) or (B, 1, H, W, D)
        else:
            noisy, clean = [x.to(device) for x in batch]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast() if amp_enabled else torch.enable_grad():
            # Forward pass
            denoised = model(noisy)

            # Composite loss (L1 + SSIM)
            l1_ssim_loss, loss_details = loss_fn(denoised, clean)

            # DreamSim loss (equal weight to L1 and SSIM)
            dreamsim_loss = torch.tensor(0.0, device=device)
            if dreamsim_fn is not None:
                try:
                    dreamsim_sim = dreamsim_fn(denoised, clean)
                    dreamsim_loss = (1.0 - dreamsim_sim).mean() * 1.0
                except Exception as e:
                    logger.debug(f"DreamSim computation failed: {e}")

            total = l1_ssim_loss + dreamsim_loss

        # Backward
        if scaler:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        # Accumulate losses
        total_loss += total.item()
        l1_loss_total += loss_details.get("l1", torch.tensor(0.0)).item()
        ssim_loss_total += loss_details.get("ssim", torch.tensor(0.0)).item()
        dreamsim_loss_total += dreamsim_loss.item()
        num_batches += 1

        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            logger.info(
                f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {total.item():.4f}"
            )

    if scheduler:
        scheduler.step()

    return {
        "total_loss": total_loss / num_batches,
        "l1_loss": l1_loss_total / num_batches,
        "ssim_loss": ssim_loss_total / num_batches,
        "dreamsim_loss": dreamsim_loss_total / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    dreamsim_fn: Optional[nn.Module],
    device: str = "cuda",
) -> Dict[str, float]:
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        if isinstance(batch, dict):
            noisy = batch["image"].to(device)
            clean = batch["label"].to(device)
        else:
            noisy, clean = [x.to(device) for x in batch]

        denoised = model(noisy)
        l1_ssim_loss, _ = loss_fn(denoised, clean)

        dreamsim_loss = torch.tensor(0.0, device=device)
        if dreamsim_fn is not None:
            try:
                dreamsim_sim = dreamsim_fn(denoised, clean)
                dreamsim_loss = (1.0 - dreamsim_sim).mean() * 1.0
            except Exception:
                pass

        total_loss += (l1_ssim_loss + dreamsim_loss).item()
        num_batches += 1

    return {"val_loss": total_loss / num_batches if num_batches > 0 else float("inf")}


def main(args):
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = (
        Path(args.output_dir) / f"{args.model}_{args.sequence}_fouRA_r{args.rank}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load IXI dataset
    from .data.ixi_loader import IXIDatasetBuilder

    builder = IXIDatasetBuilder(
        root_dir=args.ixi_root,
        sequence=args.sequence,
        num_volumes=args.num_volumes,
        spatial_dims=args.spatial_dims,
    )

    datasets = builder.create_datasets(
        cache_rate=0.5,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create FouRA model
    model = create_fouRA_model(
        model_name=args.model,
        spatial_dims=args.spatial_dims,
        rank=args.rank,
        alpha=args.alpha,
    )
    model = model.to(device)

    # Create loss and optimizer
    loss_fn, dreamsim_fn = build_loss_fn(
        spatial_dims=args.spatial_dims, device=str(device)
    )
    if dreamsim_fn:
        dreamsim_fn = dreamsim_fn.to(device)

    # Only optimize FouRA parameters
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting FouRA fine-tuning")
    logger.info("=" * 60)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            dreamsim_fn=dreamsim_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            amp_enabled=args.use_amp,
        )
        history["train"].append(train_metrics)

        # Validate
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            dreamsim_fn=dreamsim_fn,
            device=str(device),
        )
        history["val"].append(val_metrics)

        # Log
        logger.info(
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f}"
        )

        # Save checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✓ Saved best model to {checkpoint_path}")

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"✓ Saved final model to {final_path}")

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.items()},
            f,
            indent=2,
        )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained denoisers with FouRA on IXI dataset"
    )

    # Model and dataset
    parser.add_argument(
        "--model",
        default="drunet",
        help="Model name (drunet, nafnet, swinunetr, swinunetr_denoising, etc.)",
    )
    parser.add_argument(
        "--ixi-root",
        required=True,
        help="Path to IXI dataset root",
    )
    parser.add_argument(
        "--sequence",
        choices=["T1", "T2", "PD"],
        default="T1",
        help="MRI sequence type",
    )
    parser.add_argument(
        "--num-volumes",
        type=int,
        default=10,
        help="Number of volumes per sequence (default: 10)",
    )

    # Model architecture
    parser.add_argument(
        "--spatial-dims",
        type=int,
        choices=[2, 3],
        default=2,
        help="Spatial dimensions (2D or 3D)",
    )

    # FouRA settings
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="FouRA rank (default: 16)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="FouRA alpha scaling (default: 1.0)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for FouRA parameters",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="experiments/fouRA",
        help="Output directory for checkpoints and logs",
    )

    args = parser.parse_args()
    main(args)
