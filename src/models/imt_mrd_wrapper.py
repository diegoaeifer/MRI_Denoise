"""Wrapper for ImT-MRD TorchScript model.

The ImT-MRD (Implicit Transformer for MRI Reconstruction and Denoising) model
is a TorchScript model that operates on complex-valued (real + imaginary) MRI data.

The model expects:
  - Input shape: (B, 2, T, H, W) where T=1 for 2D, T≥1 for 3D
  - Channels: ch0 = real part, ch1 = imaginary part
  - Output shape: (B, 2, T, H, W) - same format as input
  - residual=True: model returns the denoised data directly (residual already applied)

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


def _find_default_weights() -> Path:
    """Find the default ImT-MRD weights file.

    Returns
    -------
    Path
        Path to the *_complex.pts file

    Raises
    ------
    FileNotFoundError
        If no *_complex.pts file is found in the weights directory
    """
    weights_dir = Path(__file__).parent.parent.parent / "weights" / "ImT-MRD"
    candidates = list(weights_dir.glob("*_complex.pts"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_complex.pts found in {weights_dir}. "
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
    """

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if model_path is None:
            model_path = _find_default_weights()
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
            If input is not 4-D or 5-D
        """
        if x.dim() == 4:
            return self._forward_2d(x)
        if x.dim() == 5:
            return self._forward_3d(x)
        raise ValueError(f"Expected 4-D or 5-D input, got {x.dim()}-D")

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Handle 2D input (B, 2, H, W) → (B, 1, H, W)."""
        real = x[:, 0:1, :, :]                          # (B, 1, H, W)
        imag = torch.zeros_like(real)                   # (B, 1, H, W)
        x_in = torch.cat([real, imag], dim=1).unsqueeze(2)  # (B, 2, 1, H, W)
        out = self.model(x_in)                          # (B, 2, 1, H, W)
        return self._magnitude(out).squeeze(2)          # (B, 1, H, W)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Handle 3D input (B, 2, D, H, W) → (B, 1, D, H, W)."""
        real = x[:, 0:1]                               # (B, 1, D, H, W)
        imag = torch.zeros_like(real)                  # (B, 1, D, H, W)
        x_in = torch.cat([real, imag], dim=1)         # (B, 2, D, H, W)
        out = self.model(x_in)                         # (B, 2, D, H, W)
        return self._magnitude(out)                    # (B, 1, D, H, W)

    @staticmethod
    def _magnitude(x: torch.Tensor) -> torch.Tensor:
        """Compute magnitude from complex tensor.

        Parameters
        ----------
        x : torch.Tensor
            Complex tensor where ch0 = real, ch1 = imaginary.
            Shape: (B, 2, ...) where ... is any spatial dimensions.

        Returns
        -------
        torch.Tensor
            Magnitude tensor of shape (B, 1, ...).
        """
        real_part = x[:, 0:1]
        imag_part = x[:, 1:2]
        return torch.sqrt(real_part ** 2 + imag_part ** 2)
