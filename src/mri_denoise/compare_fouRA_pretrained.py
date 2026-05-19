"""
Comprehensive comparison: FouRA fine-tuned vs Pretrained denoiser models.

Evaluates both models using perceptual and quantitative metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- HaarPSI (Haar Perceptual Similarity Index)
- DreamSim (perceptual quality)

Usage:
    python -m mri_denoise.compare_fouRA_pretrained \
        --model drunet \
        --ixi-root /path/to/ixi \
        --sequence T1 \
        --pretrained-ckpt pretrained.pt \
        --fouRA-ckpt fouRA_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.metrics import PSNRMetric, SSIMMetric

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_haarpsi():
    """Load HaarPSI metric if available."""
    try:
        from piq import haarpsi

        return haarpsi
    except ImportError:
        try:
            import lpips

            logger.warning("Using LPIPS as HaarPSI fallback (piq not available)")
            return lpips.LPIPS(net="vgg", verbose=False)
        except ImportError:
            logger.warning("HaarPSI not available (requires piq or lpips package)")
            return None


@torch.no_grad()
def evaluate_model_comprehensive(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    spatial_dims: int = 2,
) -> Dict[str, float]:
    """
    Comprehensive evaluation with all metrics.

    Args:
        model: Denoising model
        test_loader: Test data loader
        device: Device
        spatial_dims: 2D or 3D

    Returns:
        Dict with all metrics
    """
    from .metrics.dreamsim_metric import DreamSimMetric

    model.eval().to(device)

    # Initialize metrics
    dreamsim = DreamSimMetric(device=device)
    psnr_metric = PSNRMetric(data_range=1.0)
    ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=spatial_dims)
    haarpsi_metric = load_haarpsi()
    if haarpsi_metric:
        haarpsi_metric = haarpsi_metric.to(device).eval()

    all_psnr = []
    all_ssim = []
    all_dreamsim = []
    all_haarpsi = []

    num_batches = len(test_loader)

    for batch_idx, batch in enumerate(test_loader):
        if isinstance(batch, dict):
            noisy = batch["image"].to(device)
            clean = batch["label"].to(device)
        else:
            noisy, clean = [x.to(device) for x in batch]

        # Denoise
        denoised = model(noisy)

        # PSNR
        try:
            psnr = psnr_metric(denoised, clean)
            if psnr.ndim > 0:
                all_psnr.extend(psnr.cpu().numpy().tolist())
            else:
                all_psnr.append(psnr.item())
        except Exception as e:
            logger.debug(f"PSNR failed: {e}")

        # SSIM
        try:
            ssim = ssim_metric(denoised, clean)
            if ssim.ndim > 0:
                all_ssim.extend(ssim.cpu().numpy().tolist())
            else:
                all_ssim.append(ssim.item())
        except Exception as e:
            logger.debug(f"SSIM failed: {e}")

        # DreamSim
        try:
            dreamsim_score = dreamsim(denoised, clean)
            all_dreamsim.extend(dreamsim_score.cpu().numpy().tolist())
        except Exception as e:
            logger.debug(f"DreamSim failed: {e}")

        # HaarPSI (LPIPS)
        if haarpsi_metric is not None:
            try:
                haarpsi = haarpsi_metric(denoised, clean)
                if haarpsi.ndim > 0:
                    all_haarpsi.extend(haarpsi.cpu().detach().numpy().tolist())
                else:
                    all_haarpsi.append(haarpsi.item())
            except Exception as e:
                logger.debug(f"HaarPSI failed: {e}")

        if (batch_idx + 1) % max(1, num_batches // 5) == 0:
            logger.info(f"  Evaluated {batch_idx + 1}/{num_batches} batches")

    # Aggregate
    results = {}

    if all_psnr:
        all_psnr = np.array(all_psnr)
        results["psnr_mean"] = float(all_psnr.mean())
        results["psnr_std"] = float(all_psnr.std())
        results["psnr_median"] = float(np.median(all_psnr))

    if all_ssim:
        all_ssim = np.array(all_ssim)
        results["ssim_mean"] = float(all_ssim.mean())
        results["ssim_std"] = float(all_ssim.std())
        results["ssim_median"] = float(np.median(all_ssim))

    if all_dreamsim:
        all_dreamsim = np.array(all_dreamsim)
        results["dreamsim_mean"] = float(all_dreamsim.mean())
        results["dreamsim_std"] = float(all_dreamsim.std())
        results["dreamsim_median"] = float(np.median(all_dreamsim))

    if all_haarpsi:
        all_haarpsi = np.array(all_haarpsi)
        results["haarpsi_mean"] = float(all_haarpsi.mean())
        results["haarpsi_std"] = float(all_haarpsi.std())
        results["haarpsi_median"] = float(np.median(all_haarpsi))

    return results


def print_comparison(pretrained_results: Dict, fouRA_results: Dict) -> None:
    """Print detailed comparison between models."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE COMPARISON: FouRA Fine-tuned vs Pretrained")
    logger.info("=" * 80)

    metrics_to_compare = [
        ("psnr_mean", "PSNR (dB)", "higher"),
        ("ssim_mean", "SSIM", "higher"),
        ("dreamsim_mean", "DreamSim", "higher"),
        ("haarpsi_mean", "HaarPSI", "lower"),
    ]

    results_table = []

    for metric_name, display_name, direction in metrics_to_compare:
        if metric_name in pretrained_results and metric_name in fouRA_results:
            pre_val = pretrained_results[metric_name]
            fou_val = fouRA_results[metric_name]

            if direction == "higher":
                improvement = (
                    ((fou_val - pre_val) / abs(pre_val)) * 100 if pre_val != 0 else 0
                )
                symbol = "↑" if improvement > 0 else "↓"
                better = "FouRA" if fou_val > pre_val else "Pretrained"
            else:  # lower is better
                improvement = (
                    ((pre_val - fou_val) / abs(pre_val)) * 100 if pre_val != 0 else 0
                )
                symbol = "↓" if improvement > 0 else "↑"
                better = "FouRA" if fou_val < pre_val else "Pretrained"

            results_table.append(
                (display_name, pre_val, fou_val, improvement, symbol, better)
            )

    # Print table
    logger.info(
        "\n{:<15} {:<15} {:<15} {:<15} {:<10}".format(
            "Metric", "Pretrained", "FouRA", "Improvement", "Better"
        )
    )
    logger.info("-" * 75)

    for metric, pre_val, fou_val, improvement, symbol, better in results_table:
        improvement_str = f"{symbol} {abs(improvement):>6.2f}%"
        logger.info(
            f"{metric:<15} {pre_val:>14.4f} {fou_val:>14.4f} {improvement_str:<15} {better:<10}"
        )

    logger.info("\n" + "=" * 80)


def main(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load IXI test set
    from .data.ixi_loader import IXIDatasetBuilder

    builder = IXIDatasetBuilder(
        root_dir=args.ixi_root,
        sequence=args.sequence,
        num_volumes=args.num_volumes,
        spatial_dims=args.spatial_dims,
    )

    datasets = builder.create_datasets(cache_rate=0.0, num_workers=0)
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
    )

    if len(test_loader) == 0:
        logger.warning("No test samples available")
        return

    # Load models
    from .networks.registry import get_network

    def load_model(ckpt_path: Optional[str]) -> nn.Module:
        """Load a denoising model."""
        model = get_network(
            args.model,
            spatial_dims=args.spatial_dims,
            in_channels=2,
            out_channels=1,
        )

        if ckpt_path and Path(ckpt_path).exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            logger.info(f"✓ Loaded checkpoint: {ckpt_path}")
        else:
            logger.info("Using model without checkpoint")

        return model

    # Evaluate pretrained
    pretrained_results = None
    if args.pretrained_ckpt:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating PRETRAINED model")
        logger.info("=" * 80)
        pretrained_model = load_model(args.pretrained_ckpt)
        pretrained_results = evaluate_model_comprehensive(
            pretrained_model,
            test_loader,
            device=str(device),
            spatial_dims=args.spatial_dims,
        )

        logger.info("\nPretrained Model Results:")
        logger.info("-" * 40)
        for k, v in sorted(pretrained_results.items()):
            logger.info(f"  {k:<20}: {v:.4f}")

    # Evaluate FouRA
    fouRA_results = None
    if args.fouRA_ckpt:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating FOURA FINE-TUNED model")
        logger.info("=" * 80)
        fouRA_model = load_model(args.fouRA_ckpt)
        fouRA_results = evaluate_model_comprehensive(
            fouRA_model, test_loader, device=str(device), spatial_dims=args.spatial_dims
        )

        logger.info("\nFouRA Fine-tuned Results:")
        logger.info("-" * 40)
        for k, v in sorted(fouRA_results.items()):
            logger.info(f"  {k:<20}: {v:.4f}")

    # Compare
    if pretrained_results and fouRA_results:
        print_comparison(pretrained_results, fouRA_results)

    # Save results
    output_path = (
        Path(args.output_dir) / f"comparison_{args.model}_{args.sequence}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "sequence": args.sequence,
        "spatial_dims": args.spatial_dims,
        "pretrained": pretrained_results,
        "fouRA": fouRA_results,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of FouRA fine-tuned vs pretrained models"
    )

    # Dataset
    parser.add_argument("--ixi-root", required=True, help="Path to IXI dataset")
    parser.add_argument(
        "--sequence",
        choices=["T1", "T2", "PD"],
        default="T1",
        help="MRI sequence type",
    )
    parser.add_argument("--num-volumes", type=int, default=10)

    # Model
    parser.add_argument("--model", default="drunet", help="Model name")
    parser.add_argument("--spatial-dims", type=int, choices=[2, 3], default=2)

    # Checkpoints
    parser.add_argument("--pretrained-ckpt", help="Path to pretrained checkpoint")
    parser.add_argument("--fouRA-ckpt", help="Path to FouRA fine-tuned checkpoint")

    # Eval
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default="experiments/comparison")

    args = parser.parse_args()
    main(args)
