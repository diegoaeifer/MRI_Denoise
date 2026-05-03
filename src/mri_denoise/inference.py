"""
MONAI-native inference pipeline for MRI denoising.

Replaces src/pipeline.py. Uses SlidingWindowInferer for tiled inference
on arbitrary-size volumes, avoiding the ProcessPoolExecutor bug in pipeline.py
(model cannot be pickled across processes).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)

logger = logging.getLogger(__name__)


def build_inference_transforms(cfg: Dict[str, Any]) -> Compose:
    """Minimal preprocessing for inference (no augmentation, no noise injection)."""
    data_cfg = cfg.get("data", {})
    percentile_lower = data_cfg.get("percentile_lower", 1.0)
    percentile_upper = data_cfg.get("percentile_upper", 99.0)

    return Compose(
        [
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=percentile_lower,
                upper=percentile_upper,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            ToTensord(keys="image"),
        ]
    )


def run_inference(
    network: nn.Module,
    image_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    cfg: Dict[str, Any],
    device: torch.device,
    sigma: float = 0.1,
    batch_size: int = 1,
) -> List[Path]:
    """
    Run sliding-window inference on a list of images and save results.

    Args:
        network: Trained denoising network (expects 2-channel input).
        image_paths: List of paths to input NIfTI / DICOM files.
        output_dir: Directory to write denoised outputs.
        cfg: Pipeline config dict.
        device: Torch device.
        sigma: Noise level for sigma_map channel (uniform scalar fallback
               when no sigma_map is available at inference time).
        batch_size: Number of images per batch.

    Returns:
        List of output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    roi_size = tuple(data_cfg.get("image_size", [256, 256]))
    sw_batch = data_cfg.get("sw_batch_size", 4)
    overlap = data_cfg.get("sw_overlap", 0.25)

    transforms = build_inference_transforms(cfg)
    datalist = [{"image": str(p)} for p in image_paths]
    dataset = CacheDataset(data=datalist, transform=transforms, cache_rate=0.0)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch,
        overlap=overlap,
        mode="gaussian",
    )

    network.eval()
    output_paths: List[Path] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = batch["image"].to(device)  # (B, 1, ...)

            # Build 2-channel input: [noisy_image | sigma_map]
            sigma_map = torch.full_like(img, sigma)
            model_input = torch.cat([img, sigma_map], dim=1)  # (B, 2, ...)

            pred = inferer(model_input, network)
            pred = torch.clamp(pred, 0.0, 1.0)

            for b in range(pred.shape[0]):
                global_idx = i * batch_size + b
                src_path = Path(image_paths[global_idx])
                out_name = src_path.stem + "_denoised" + src_path.suffix
                out_path = output_dir / out_name
                _save_tensor(pred[b], out_path, batch["image_meta_dict"], b)
                output_paths.append(out_path)
                logger.info(f"Saved denoised image → {out_path}")

    return output_paths


def _save_tensor(
    tensor: torch.Tensor,
    out_path: Path,
    meta_dict: Optional[Dict[str, Any]],
    batch_idx: int,
) -> None:
    """Save a single (C, ...) tensor as NIfTI using nibabel."""
    try:
        import nibabel as nib
        import numpy as np

        arr = tensor.cpu().numpy()
        if arr.shape[0] == 1:
            arr = arr[0]  # remove channel dim for grayscale

        # Try to recover affine from MONAI meta_dict
        affine = None
        if meta_dict is not None and "affine" in meta_dict:
            aff = meta_dict["affine"]
            if isinstance(aff, torch.Tensor):
                aff = aff[batch_idx] if aff.ndim == 3 else aff
                affine = aff.cpu().numpy()

        if affine is None:
            affine = np.eye(4)

        nii = nib.Nifti1Image(arr.astype(np.float32), affine=affine)
        nib.save(nii, str(out_path))
    except ImportError:
        logger.warning("nibabel not available; skipping NIfTI save for %s", out_path)
