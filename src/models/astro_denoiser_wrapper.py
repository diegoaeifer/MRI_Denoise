"""Wrapper for astro_denoiser_N2N_N2S (andriymelnykov/astro_denoiser_N2N_N2S).

The upstream repository is a Keras/TensorFlow Colab notebook with no PyTorch
code.  This wrapper provides a **native PyTorch reimplementation** of the
DnCNN-style residual architecture defined in that notebook ('build_train16'),
so the model integrates cleanly with the MRI denoising factory without
requiring TensorFlow.

Architecture (matches `build_train16` from the notebook):
  Conv(7×7, 32 filters, ReLU) → 6× Conv(7×7, 32 filters, ReLU)
  → Conv(3×3, out_channels) → residual subtract

Adapts (B, 2, H, W) input (noisy image + sigma map) → (B, 1, H, W) output.

Forward convention:
  ch0 → noisy image  (input to the residual network)
  ch1 → sigma map    (not used by this architecture; ignored)
"""

import torch
import torch.nn as nn


class _ResidualDnCNN(nn.Module):
    """Pure-PyTorch DnCNN-style residual denoiser.

    Mirrors the Keras `build_train16` architecture from the
    andriymelnykov/astro_denoiser_N2N_N2S notebook:
      - 1 entry conv (kernel_size=7, `filters` channels, ReLU)
      - `depth` hidden convs (kernel_size=7, `filters` channels, ReLU)
      - 1 exit conv (kernel_size=3, `out_channels` channels, no activation)
      - residual subtraction: output = input − predicted_noise
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        filters: int = 32,
        depth: int = 6,
    ):
        super().__init__()
        layers = []
        # Entry conv
        layers.append(nn.Conv2d(in_channels, filters, kernel_size=7, padding=3, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Hidden convs
        for _ in range(depth):
            layers.append(nn.Conv2d(filters, filters, kernel_size=7, padding=3, bias=True))
            layers.append(nn.ReLU(inplace=True))
        # Exit conv (predicts residual / noise)
        layers.append(nn.Conv2d(filters, out_channels, kernel_size=3, padding=1, bias=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.body(x)
        return x - noise   # residual connection


class AstroDenoiserWrapper(nn.Module):
    """Factory-compatible wrapper around _ResidualDnCNN.

    Accepts (B, 2, H, W) and returns (B, 1, H, W).

    Parameters
    ----------
    in_channels:
        Must be 2 (image + sigma map).  Kept for API symmetry.
    filters:
        Number of convolutional filters (matches `f=32` in build_train16).
    depth:
        Number of hidden convolutional layers (matches `depth=6` in build_train16).
    """

    def __init__(
        self,
        in_channels: int = 2,
        filters: int = 32,
        depth: int = 6,
    ):
        super().__init__()
        # Internal network always operates on 1-channel images
        self.net = _ResidualDnCNN(in_channels=1, out_channels=1, filters=filters, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, H, W)
            ch0 = noisy image, ch1 = sigma map (ignored by this model)

        Returns
        -------
        (B, 1, H, W) denoised image
        """
        img = x[:, :1, :, :]   # (B, 1, H, W)
        return self.net(img)
