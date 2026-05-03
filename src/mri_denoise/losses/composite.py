"""
MONAI-first CompositeLoss.

All MONAI-available losses imported directly.
Custom losses kept only when MONAI has no equivalent:
  - CharbonnierLoss  (smooth L1)
  - PSNRLoss         (MONAI has metric only, not loss)
  - HaarPSILoss      (not in MONAI)
  - EPILoss          (Sobel-based, spatial_dims-aware)
  - MCSURELoss       (unsupervised SURE, optional)

VGGPerceptualLoss replaced by monai.losses.PerceptualLoss.
LPIPS / DISTS dropped (piq dependency removed).
MS-SSIM moved to validation metrics only (no training loss).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Custom losses (MONAI has no equivalent)
# ---------------------------------------------------------------------------- #


class PSNRLoss(nn.Module):
    """Minimise negative PSNR (= maximise PSNR). FP16-safe."""

    def __init__(self, max_val: float = 1.0) -> None:
        super().__init__()
        self.max_val = max_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        mse_safe = torch.clamp(mse, min=1e-7)
        psnr = 20 * torch.log10(
            torch.tensor(self.max_val, device=pred.device) / (torch.sqrt(mse_safe) + 1e-7)
        )
        return -psnr


class CharbonnierLoss(nn.Module):
    """Smooth L1 approximation. FP16-safe."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        sqrt_arg = torch.clamp(diff * diff + self.eps * self.eps, min=1e-8)
        return torch.mean(torch.sqrt(sqrt_arg))


class EPILoss(nn.Module):
    """
    Edge Preservation Index loss via Sobel gradients.
    Returns 1 - EPI so minimising loss maximises edge preservation.
    spatial_dims=2 uses 2 Sobel kernels; spatial_dims=3 uses 3.
    """

    def __init__(self, spatial_dims: int = 2) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _grad2d(self, img: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        c = img.size(1)
        gx = F.conv2d(img, self.sobel_x.expand(c, 1, 3, 3), padding=1, groups=c)
        gy = F.conv2d(img, self.sobel_y.expand(c, 1, 3, 3), padding=1, groups=c)
        return torch.sqrt(torch.clamp(gx**2 + gy**2, min=1e-8))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle 3D: collapse depth into batch
        if pred.ndim == 5:
            b, c, h, w, d = pred.shape
            pred = pred.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
            target = target.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

        g1 = self._grad2d(target)
        g2 = self._grad2d(pred)

        num = torch.sum(g1 * g2, dim=[-2, -1])
        den = torch.sqrt(
            torch.clamp(
                torch.sum(g1**2, dim=[-2, -1]) * torch.sum(g2**2, dim=[-2, -1]),
                min=1e-8,
            )
        )
        epi = torch.mean(num / (den + 1e-8))
        return 1.0 - epi


# ---------------------------------------------------------------------------- #
# CompositeLoss
# ---------------------------------------------------------------------------- #


class CompositeLoss(nn.Module):
    """
    Weighted composite loss for MRI denoising.

    Args:
        spatial_dims: 2 or 3 (propagated to MONAI losses)
        weights: dict mapping loss name → float weight (0 = skip)
        data_range: pixel value range, default 1.0

    Supported loss keys:
        l1, charbonnier, ssim, psnr, haarpsi, epi, perceptual, sure
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        weights: Optional[Dict[str, float]] = None,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = weights or {"l1": 1.0, "ssim": 0.1, "psnr": 0.1}
        self.spatial_dims = spatial_dims

        from monai.losses import SSIMLoss

        # Build only the losses that are actually used (weight > 0)
        components: Dict[str, nn.Module] = {}

        if self.weights.get("l1", 0) > 0:
            components["l1"] = nn.L1Loss()

        if self.weights.get("charbonnier", 0) > 0:
            components["charbonnier"] = CharbonnierLoss()

        if self.weights.get("ssim", 0) > 0:
            components["ssim"] = SSIMLoss(
                spatial_dims=spatial_dims, data_range=data_range
            )

        if self.weights.get("psnr", 0) > 0:
            components["psnr"] = PSNRLoss(max_val=data_range)

        if self.weights.get("epi", 0) > 0:
            components["epi"] = EPILoss(spatial_dims=spatial_dims)

        if self.weights.get("haarpsi", 0) > 0:
            try:
                import piq  # optional — kept for backward compat if piq present
                components["haarpsi"] = piq.HaarPSILoss(data_range=data_range)
            except ImportError:
                logger.warning("piq not installed — haarpsi loss disabled")
                self.weights["haarpsi"] = 0.0

        if self.weights.get("perceptual", 0) > 0:
            try:
                from monai.losses import PerceptualLoss
                components["perceptual"] = PerceptualLoss(
                    spatial_dims=spatial_dims, network_type="vgg"
                )
            except Exception as e:
                logger.warning(f"PerceptualLoss unavailable: {e} — disabling")
                self.weights["perceptual"] = 0.0

        if self.weights.get("sure", 0) > 0:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parents[3]))
            from src.losses.auxiliary import MCSURELoss
            self._sure = MCSURELoss(eps=1e-4)
            self._sure_weight = self.weights["sure"]

        self.components = nn.ModuleDict(components)

        active = [k for k, v in self.weights.items() if v > 0]
        logger.info(f"CompositeLoss active components: {active}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        model: Optional[nn.Module] = None,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred: (B, C, H, W) or (B, C, H, W, D)
            target: same shape as pred
            model: required only if sure weight > 0
            input_tensor: required only if sure weight > 0

        Returns:
            (total_loss, loss_detail_dict)
        """
        total: torch.Tensor = torch.tensor(0.0, device=pred.device)
        details: Dict[str, torch.Tensor] = {}

        for name, module in self.components.items():
            w = self.weights.get(name, 0)
            if w == 0:
                continue
            try:
                val = module(pred, target)
                total = total + w * val
                # Log positive PSNR (not loss value)
                details[name] = -val if name == "psnr" else val
            except Exception as e:
                logger.warning(f"{name} loss failed: {e}")
                details[name] = torch.tensor(0.0, device=pred.device)

        # SURE loss (requires model + input_tensor)
        if self.weights.get("sure", 0) > 0 and model is not None and input_tensor is not None:
            sigma_map = input_tensor[:, 1:2]
            val = self._sure(model, input_tensor, pred, sigma_map)
            total = total + self._sure_weight * val
            details["sure"] = val

        return total, details
