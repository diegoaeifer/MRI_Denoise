"""Wrapper for ImT-MRD TorchScript model.

The ImT-MRD (Implicit Transformer for MRI Reconstruction and Denoising) model
is a TorchScript model that operates on complex-valued (real + imaginary) MRI data.

The model expects:
  - Input shape: (B, T, 2, H, W) where T=1 for 2D, T≥1 for 3D
  - dim-2 channels: ch0 = real part, ch1 = imaginary part
  - Output shape: (B, T, 2, H, W) - same format as input

This wrapper adapts the factory convention:
  - Input: (B, 2, H, W) for 2D or (B, 2, D, H, W) for 3D
  - ch0 = magnitude image (used as real part)
  - ch1 = sigma map (ignored; zero padding used for imaginary part)
  - Output: (B, 1, H, W) or (B, 1, D, H, W) - magnitude of denoised complex data
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def _find_default_weights(variant: str = "complex") -> Path:
    """Find ImT-MRD weights matching the requested variant.

    Parameters
    ----------
    variant : 'complex' or 'residual'
    """
    weights_dir = Path(__file__).parent.parent.parent / "weights" / "ImT-MRD"
    pattern = f"*_{variant}.pts"
    candidates = list(weights_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {pattern} found in {weights_dir}. "
            "Download from https://github.com/zherenz/ImT-MRD"
        )
    return candidates[0]


class ImtMrdWrapper(nn.Module):
    """Wrapper for ImT-MRD TorchScript model.

    Converts factory-standard input (B, 2, H, W) or (B, 2, D, H, W) to
    TorchScript input (B, 2, T, H, W) and returns (B, 1, H, W) or (B, 1, D, H, W).

    Parameters
    ----------
    model_path : str, os.PathLike, or None
        Path to the TorchScript model file (*_complex.pts).
        If None, searches for weights in weights/ImT-MRD/.
    freeze_backbone : bool
        If True, disable gradient computation for model parameters.

    Notes
    -----
    To move the model to GPU, call `model.model.to(device)` directly.
    """

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        freeze_backbone: bool = True,
        model_variant: str = "complex",
    ) -> None:
        super().__init__()
        if model_path is None:
            model_path = _find_default_weights(variant=model_variant)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ImT-MRD weights not found at {model_path}. "
                "Download from https://github.com/zherenz/ImT-MRD"
            )
        self.model = torch.jit.load(str(model_path), map_location="cpu")
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 2, H, W) or (B, 2, D, H, W).
            ch0 = magnitude image, ch1 = sigma map (ignored)

        Returns
        -------
        torch.Tensor
            Denoised magnitude output of shape (B, 1, H, W) or (B, 1, D, H, W)

        Raises
        ------
        ValueError
            If input is not 4-D or 5-D, or if channel dimension is not 2
        """
        if x.dim() not in (4, 5):
            raise ValueError(f"Expected 4-D or 5-D input, got {x.dim()}-D")
        if x.shape[1] != 2:
            raise ValueError(f"Expected 2 input channels, got {x.shape[1]}")
        if x.dim() == 4:
            return self._forward_2d(x)
        return self._forward_3d(x)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Handle 2D input (B, 2, H, W) → (B, 1, H, W)."""
        B, _, H, W = x.shape
        real = x[:, 0:1, :, :]          # (B, 1, H, W)
        imag = torch.zeros_like(real)
        # Model expects (B, T, C, H, W) = (B, 1, 2, H, W)
        x_in = torch.stack([real, imag], dim=2)  # (B, 1, 2, H, W)
        out = self.model(x_in)           # (B, 1, 2, H, W)
        r, i = out[:, :, 0], out[:, :, 1]  # (B, 1, H, W) each
        return torch.hypot(r, i)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Handle 3D input (B, 2, D, H, W) → (B, 1, D, H, W)."""
        real = x[:, 0:1]   # (B, 1, D, H, W)
        imag = torch.zeros_like(real)
        # Model expects (B, T, C, H, W) = (B, D, 2, H, W)
        # real/imag are (B, 1, D, H, W) → squeeze chan → (B, D, H, W) → stack
        r = real.squeeze(1)   # (B, D, H, W)
        im = imag.squeeze(1)  # (B, D, H, W)
        x_in = torch.stack([r, im], dim=2)  # (B, D, 2, H, W)
        out = self.model(x_in)              # (B, D, 2, H, W)
        mag = torch.hypot(out[:, :, 0], out[:, :, 1])  # (B, D, H, W)
        return mag.unsqueeze(1)  # (B, 1, D, H, W)

