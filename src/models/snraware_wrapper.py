"""Wrapper for SNRAware TorchScript MRI denoising models.

SNRAware expects a 3-channel input [C=3, T, H, W]:
  ch0 = real part (or magnitude for magnitude-only input)
  ch1 = imaginary part (zeros when no phase available)
  ch2 = g-factor map (ones when no parallel-imaging map available)
Output: [C=2, T, H, W] — denoised real and imaginary channels.

This wrapper adapts the factory convention:
  Input:  (B, 2, H, W)    for 2D — ch0=magnitude, ch1=sigma map (ignored)
          (B, 2, D, H, W) for 3D
  Output: (B, 1, H, W) or (B, 1, D, H, W) — denoised magnitude

Tiled patch inference (cutout/overlap) is handled by SNRAware-Private's
apply_model, which pads, patches, and reconstructs with weighted blending.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_SNRAWARE_SRC = Path(__file__).resolve().parents[3] / "SNRAware-Private" / "src"
if str(_SNRAWARE_SRC) not in sys.path:
    sys.path.insert(0, str(_SNRAWARE_SRC))

from snraware.projects.mri.denoising.inference import apply_model  # noqa: E402

_WEIGHTS_BASE = Path(__file__).resolve().parents[2] / "weights" / "SNRAware"


class SNRAwareWrapper(nn.Module):
    """Wraps an SNRAware TorchScript model with tiled patch inference.

    Parameters
    ----------
    model_size : 'small', 'medium', or 'large'
    cutout : (H, W, T) patch size — must match training config (default 64x64x16)
    overlap : (H, W, T) overlap per axis — must be strictly less than cutout
    """

    def __init__(
        self,
        model_size: str = "small",
        cutout: tuple[int, int, int] = (64, 64, 16),
        overlap: tuple[int, int, int] = (16, 16, 8),
    ) -> None:
        super().__init__()
        pts = _WEIGHTS_BASE / model_size / f"snraware_{model_size}_model.pts"
        if not pts.exists():
            raise FileNotFoundError(
                f"SNRAware weights not found: {pts}\n"
                "Download from https://github.com/MLI-lab/SNRAware"
            )
        self.model = torch.jit.load(str(pts), map_location="cpu")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cutout = cutout
        self.overlap = overlap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run denoising with tiled-patch inference.

        Parameters
        ----------
        x : (B, 2, H, W) or (B, 2, D, H, W)
            ch0 = magnitude image [0, 1]
            ch1 = sigma map (ignored; gmap=ones is used)

        Returns
        -------
        (B, 1, H, W) or (B, 1, D, H, W)  denoised magnitude
        """
        if x.dim() == 4:
            return self._forward_batch(x, is_3d=False)
        if x.dim() == 5:
            return self._forward_batch(x, is_3d=True)
        raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")

    def _forward_batch(self, x: torch.Tensor, is_3d: bool) -> torch.Tensor:
        device_str = str(x.device)
        outs: list[torch.Tensor] = []

        for b in range(x.shape[0]):
            mag_np = x[b, 0].cpu().numpy().astype(np.float32)

            if is_3d:
                # (D, H, W) → (H, W, D) — apply_model expects [H, W, T]
                data = mag_np.transpose(1, 2, 0)
            else:
                # (H, W) → (H, W, T=1)
                data = mag_np[:, :, np.newaxis]

            gmap = np.ones_like(data)
            result = apply_model(
                self.model,
                data,
                gmap,
                cutout=self.cutout,
                overlap=self.overlap,
                batch_size=1,
                device=device_str,
            )  # returns (H, W, T) magnitude float32

            if is_3d:
                # (H, W, D) → (D, H, W) → add channel → (1, D, H, W)
                arr = result.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            else:
                # (H, W, 1) → drop T → (H, W) → add channel → (1, H, W)
                arr = result[:, :, 0][np.newaxis].astype(np.float32)

            outs.append(torch.from_numpy(arr))

        return torch.stack(outs, dim=0)  # (B, 1, H, W) or (B, 1, D, H, W)
