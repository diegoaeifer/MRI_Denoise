"""Wrapper for ImT-MRD TorchScript models.

Two available weights and their expected input formats:

  image_residual (*_residual.pts) — trained on magnitude images:
    Input:  (B, T, 1, H, W)  — 1 magnitude channel
    Output: (B, T, 1, H, W)  — denoised magnitude
    Use this for magnitude-only MRI (no phase data available).

  noGmap_complex (*_complex.pts) — trained on complex MRI without gmap:
    Input:  (B, T, 2, H, W)  — 2 channels: real + imaginary
    Output: (B, T, 2, H, W)  — denoised real + imaginary
    For magnitude input: real=mag, imag=zeros (slight domain mismatch).

This wrapper adapts the factory convention:
  Input:  (B, 2, H, W)    for 2D — ch0=magnitude, ch1=sigma map (ignored)
          (B, 2, D, H, W) for 3D
  Output: (B, 1, H, W) or (B, 1, D, H, W)

Default variant is 'residual' (image_residual weights) — best for magnitude-only MRI.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def _find_default_weights(variant: str = "residual") -> Path:
    """Find ImT-MRD weights matching the requested variant.

    Parameters
    ----------
    variant : 'residual' (image_residual, magnitude-only) or 'complex' (noGmap_complex)
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
    the TorchScript input format and returns (B, 1, H, W) or (B, 1, D, H, W).

    Parameters
    ----------
    model_path : str, os.PathLike, or None
        Path to the TorchScript model file. If None, searches in weights/ImT-MRD/.
    freeze_backbone : bool
        If True, disable gradient computation for model parameters.
    model_variant : 'residual' or 'complex'
        'residual' — image_residual weights, expects 1-channel magnitude (recommended).
        'complex' — noGmap_complex weights, expects 2-channel real+imag input.
    input_scale : float
        Additional scaling factor applied on top of power normalization.
        Sweeps denoising strength around the calibrated base. Default 1.0 = no extra scale.
    power_norm : float
        Training power normalization constant from the model config (power_norm=1600.0).
        Input is multiplied by sqrt(power_norm) before inference and divided after.
        This matches the training distribution. Default 1600.0.
    """

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        freeze_backbone: bool = True,
        model_variant: str = "residual",
        input_scale: float = 1.0,
        power_norm: float = 1600.0,
    ) -> None:
        super().__init__()
        import math
        self.is_complex = model_variant == "complex"
        self.input_scale = input_scale
        self.power_scale = math.sqrt(power_norm)  # sqrt(1600) = 40.0
        if model_path is None:
            model_path = _find_default_weights(variant=model_variant)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ImT-MRD weights not found at {model_path}. "
                "Download from https://github.com/zherenz/ImT-MRD"
            )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(str(model_path), map_location=self._device)
        self.model = self.model.to(self._device)
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
        if x.dim() not in (4, 5):
            raise ValueError(f"Expected 4-D or 5-D input, got {x.dim()}-D")
        if x.dim() == 4:
            return self._forward_2d(x)
        return self._forward_3d(x)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Handle 2D input (B, 2, H, W) → (B, 1, H, W). Uses T=1 (no temporal context)."""
        effective_scale = self.power_scale * self.input_scale
        mag = x[:, 0:1] * effective_scale  # (B, 1, H, W)
        if self.is_complex:
            x_in = torch.stack([mag, torch.zeros_like(mag)], dim=2).to(self._device)
            out = self.model(x_in)
            return torch.hypot(out[:, 0, 0], out[:, 0, 1]).unsqueeze(1) / effective_scale
        else:
            x_in = mag.unsqueeze(1).to(self._device)   # (B, 1, 1, H, W)
            out = self.model(x_in)                      # (B, 1, 1, H, W)
            return out[:, 0] / effective_scale           # (B, 1, H, W)

    def forward_stack(self, stack: torch.Tensor) -> torch.Tensor:
        """Denoise a temporal stack of 2D slices, return centre slice.

        Parameters
        ----------
        stack : torch.Tensor  shape (B, T, H, W) — T consecutive magnitude slices.

        Returns
        -------
        torch.Tensor  shape (B, 1, H, W) — denoised centre slice.
        """
        effective_scale = self.power_scale * self.input_scale
        mag = stack * effective_scale   # (B, T, H, W)
        if self.is_complex:
            zeros = torch.zeros_like(mag)
            x_in = torch.stack([mag, zeros], dim=2).to(self._device)  # (B, T, 2, H, W)
            out = self.model(x_in)  # (B, T, 2, H, W)
            mag_out = torch.hypot(out[:, :, 0], out[:, :, 1])          # (B, T, H, W)
        else:
            x_in = mag.unsqueeze(2).to(self._device)  # (B, T, 1, H, W)
            out = self.model(x_in)                     # (B, T, 1, H, W)
            mag_out = out[:, :, 0]                     # (B, T, H, W)
        mid = stack.shape[1] // 2
        return mag_out[:, mid:mid+1] / effective_scale  # (B, 1, H, W)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Handle 3D input (B, 2, D, H, W) → (B, 1, D, H, W)."""
        effective_scale = self.power_scale * self.input_scale
        mag = x[:, 0] * effective_scale  # (B, D, H, W)
        if self.is_complex:
            # noGmap_complex: (B, T=D, 2, H, W) — real=mag, imag=zeros
            zeros = torch.zeros_like(mag)
            x_in = torch.stack([mag, zeros], dim=2).to(self._device)  # (B, D, 2, H, W)
            out = self.model(x_in)                   # (B, D, 2, H, W)
            mag_out = torch.hypot(out[:, :, 0], out[:, :, 1])  # (B, D, H, W)
            return mag_out.unsqueeze(1) / effective_scale       # (B, 1, D, H, W)
        else:
            # image_residual: (B, T=D, 1, H, W) — magnitude only
            x_in = mag.unsqueeze(2).to(self._device)  # (B, D, 1, H, W)
            out = self.model(x_in)                     # (B, D, 1, H, W)
            return out.permute(0, 2, 1, 3, 4) / effective_scale  # (B, 1, D, H, W)

