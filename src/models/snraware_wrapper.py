"""Wrapper for SNRAware (IFM MRI Denoising) TorchScript model.

The SNRAware model is a 3D complex-MRI denoiser trained by the IFM group.

Model contract:
  - Input:  (B, 3, T, H, W)   — channels: [real, imaginary, g-factor map]
  - Output: (B, 2, T, H, W)   — channels: [real, imaginary] of denoised image
  - Spatial size must be exactly PATCH_SIZE × PATCH_SIZE (default 64×64)

This wrapper adapts the factory (B, 2, H, W) convention:
  - ch0 = magnitude image  → used as real part; imaginary set to zero
  - ch1 = sigma map        → ignored; g-factor map set to ones
  - Output: (B, 1, H, W)  magnitude of denoised complex image

For images larger than PATCH_SIZE, tiled inference is performed with a
Hann-window blending to suppress seam artefacts.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_WEIGHTS_ROOT = Path(__file__).parent.parent.parent / "weights" / "SNRAware"
_MODEL_SIZES = ("small", "medium", "large")

PATCH_SIZE: int = 64


def _find_weights(model_path: str | os.PathLike | None,
                  model_size: str = "medium") -> Path:
    if model_path is not None:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"SNRAware weights not found at {p}")
        return p
    size_dir = _WEIGHTS_ROOT / model_size
    exact = size_dir / f"snraware_{model_size}_model.pts"
    if exact.exists():
        return exact
    candidates = list(size_dir.glob("*.pts"))
    if not candidates:
        raise FileNotFoundError(
            f"SNRAware {model_size} weights not found in {size_dir}. "
            "Download from https://github.com/MLI-lab/SNRAware"
        )
    return candidates[0]


def _hann_window_2d(size: int, device: torch.device) -> torch.Tensor:
    """2D Hann window for patch blending."""
    w1d = torch.hann_window(size, device=device)
    return w1d.unsqueeze(0) * w1d.unsqueeze(1)


class SNRAwareWrapper(nn.Module):
    """Tiled-inference wrapper for the SNRAware TorchScript MRI denoiser.

    Parameters
    ----------
    model_path : str, os.PathLike, or None
        Path to *snraware_small_model.pts*.  If None, uses the default location.
    overlap : int
        Pixel overlap between adjacent 64×64 tiles (default 16).
    freeze : bool
        Freeze backbone parameters (recommended: True).
    """

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        model_size: str = "medium",
        overlap: int = 32,
        freeze: bool = True,
        use_sigma_as_gmap: bool = False,
    ) -> None:
        super().__init__()
        if model_size not in _MODEL_SIZES:
            raise ValueError(
                f"model_size must be one of {_MODEL_SIZES}, got {model_size!r}"
            )
        self.model_size = model_size
        self.overlap = overlap
        self.use_sigma_as_gmap = use_sigma_as_gmap
        weights = _find_weights(model_path, model_size)
        self.model = torch.jit.load(str(weights), map_location="cpu")
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a 2D batch.

        Parameters
        ----------
        x : (B, 2, H, W)
            ch0 = magnitude image, ch1 = sigma map (ignored by SNRAware).

        Returns
        -------
        (B, 1, H, W) denoised magnitude.
        """
        if x.dim() != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected (B, 2, H, W), got {tuple(x.shape)}")

        B, _, H, W = x.shape
        device = x.device
        out = torch.zeros(B, 1, H, W, device=device, dtype=x.dtype)

        for b in range(B):
            mag = x[b, 0]          # (H, W)
            sigma_map = x[b, 1]    # (H, W)
            out[b, 0] = self._tile_predict(mag, sigma_map, device)

        return out

    @torch.no_grad()
    def _tile_predict(self, mag: torch.Tensor, sigma_map: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        """Run tiled inference on a single 2D image.

        Parameters
        ----------
        mag : (H, W) float32 — magnitude image.
        sigma_map : (H, W) float32 — per-pixel sigma map.

        Returns
        -------
        (H, W) float32 — denoised magnitude.
        """
        H, W = mag.shape
        P = PATCH_SIZE
        stride = max(1, P - self.overlap)

        # Global sigma mean for relative gmap normalisation
        sigma_global_mean = sigma_map.mean().clamp(min=1e-8) if self.use_sigma_as_gmap else None

        # Pad so each dimension is covered by at least one patch
        pad_h = max(0, P - H) if H <= P else (stride - (H - P) % stride) % stride
        pad_w = max(0, P - W) if W <= P else (stride - (W - P) % stride) % stride
        if pad_h or pad_w:
            # F.pad needs at least 3D; unsqueeze/squeeze around the call
            mag = F.pad(mag.unsqueeze(0).unsqueeze(0),
                        (0, pad_w, 0, pad_h), mode="reflect").squeeze(0).squeeze(0)
            if self.use_sigma_as_gmap:
                sigma_map = F.pad(sigma_map.unsqueeze(0).unsqueeze(0),
                                  (0, pad_w, 0, pad_h), mode="reflect").squeeze(0).squeeze(0)
        pH, pW = mag.shape

        acc = torch.zeros(pH, pW, device=device)
        wgt = torch.zeros(pH, pW, device=device)
        win = _hann_window_2d(P, device)

        for y in range(0, pH - P + 1, stride):
            for x_off in range(0, pW - P + 1, stride):
                patch = mag[y : y + P, x_off : x_off + P]  # (P, P)
                # Build (1, 3, 1, P, P): real=patch, imag=0, gmap
                real = patch.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,P,P)
                imag = torch.zeros_like(real)
                if self.use_sigma_as_gmap:
                    sigma_patch = sigma_map[y : y + P, x_off : x_off + P]
                    gmap_vals = (sigma_patch / sigma_global_mean).clamp(min=0.1, max=10.0)
                    gmap = gmap_vals.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,P,P)
                else:
                    gmap = torch.ones_like(real)
                inp = torch.cat([real, imag, gmap], dim=1)  # (1,3,1,P,P)
                inp = inp.to(next(self.model.parameters()).device)
                denoised_5d = self.model(inp)  # (1,2,1,P,P)
                # Magnitude from complex output
                r = denoised_5d[0, 0, 0]  # (P,P)
                i = denoised_5d[0, 1, 0]  # (P,P)
                mag_out = torch.hypot(r, i).to(device)
                acc[y : y + P, x_off : x_off + P] += mag_out * win
                wgt[y : y + P, x_off : x_off + P] += win

        result = acc / wgt.clamp(min=1e-8)
        # Crop back to original size
        return result[:H, :W]
