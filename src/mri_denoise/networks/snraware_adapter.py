"""
SNRAware adapter for the MONAI denoising registry.

Bridges SNRAware's DenoisingModel (which expects [B, C, T, H, W] complex tensors
with a g-factor map) to the MONAI pipeline's 2-channel (image + sigma_map) format.

The key insight from PR #70: sigma_map plays the same role as gmap in SNRAware —
both encode spatially-varying noise/sensitivity information as a second channel.

This file lives in the mri_denoise package to maintain the MONAI-first principle;
it wraps the same DenoisingModel as src/models/snraware_adapter.py but exposes the
standard (in_channels, out_channels, spatial_dims) constructor interface.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SNRAwareDenoisingAdapter(nn.Module):
    """
    Wraps SNRAware DenoisingModel for the (image + sigma_map) → denoised pipeline.

    Input:  (B, 2, H, W)       or (B, 2, H, W, D)  — image + sigma_map
    Output: (B, 1, H, W)       or (B, 1, H, W, D)  — denoised magnitude

    SNRAware expects [B, C, T, H, W] where T is a depth / time dimension.
    For 2D inputs T=1 is inserted; for 3D D becomes T.

    Args:
        spatial_dims: 2 or 3
        in_channels: always 2 (image + sigma_map)
        out_channels: always 1 (denoised)
        snraware_config: OmegaConf config dict for DenoisingModel backbone.
            If None, a compact default UNet config is used (suitable for fine-tuning).
        patch_depth: SNRAware T dimension (only used for 3D; ignored for 2D)
        num_channels: backbone channel width (default 32 for lightweight default)
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 2,
        out_channels: int = 1,
        snraware_config: Optional[dict] = None,
        patch_depth: int = 16,
        num_channels: int = 32,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims

        try:
            from snraware.projects.mri.denoising.model import DenoisingModel
            from omegaconf import OmegaConf

            cfg = snraware_config or {
                "backbone": {
                    "name": "unet",
                    "num_of_channels": num_channels,
                    "block_str": ["T1L1G1", "T1L1G1"],
                    "block": {
                        "cell": {
                            "window_size": [8, 8, 16],
                            "patch_size": [4, 4, 2],
                            "norm_mode": "layer",
                            "n_head": num_channels,
                        }
                    },
                }
            }

            D = patch_depth if spatial_dims == 3 else 1
            self._model = DenoisingModel(
                config=OmegaConf.create(cfg),
                D=D,
                H=64,   # default; SNRAware re-creates for each forward
                W=64,
                C_in=in_channels,
                C_out=out_channels,
            )
            self._available = True
            logger.info("SNRAwareDenoisingAdapter: DenoisingModel loaded successfully")

        except ImportError:
            logger.warning(
                "snraware or omegaconf not installed. "
                "SNRAwareDenoisingAdapter will raise at forward() time. "
                "Install 'snraware' from Microsoft to use this model."
            )
            self._model = None
            self._available = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._available or self._model is None:
            raise RuntimeError(
                "SNRAware is not installed. "
                "pip install snraware to use SNRAwareDenoisingAdapter."
            )

        # Reshape to SNRAware's [B, C, T, H, W] format
        if x.ndim == 4:
            # (B, 2, H, W) → (B, 2, 1, H, W)
            x = x.unsqueeze(2)
            out = self._model(x)
            return out.squeeze(2)              # (B, 1, H, W)

        elif x.ndim == 5:
            # (B, 2, H, W, D) → (B, 2, D, H, W)
            x = x.permute(0, 1, 4, 2, 3)
            out = self._model(x)
            return out.permute(0, 1, 3, 4, 2)  # (B, 1, H, W, D)

        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.ndim}D")
