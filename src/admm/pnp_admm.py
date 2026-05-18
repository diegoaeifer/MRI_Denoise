"""
Plug-and-Play ADMM for MRI denoising.

Algorithm (Venkatakrishnan et al., 2013):
  x-update: x = denoiser(z - u, sigma_denoiser)
  z-update: z = prox_{rho * regularizer}(x + u)
  u-update: u = u + x - z

Convergence criterion: ||x_k - x_{k-1}|| / (||x_{k-1}|| + eps) < tol
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .regularizers import Regularizer, ElasticNetRegularizer


def pnp_admm_denoise(
    noisy: torch.Tensor,
    denoiser: nn.Module,
    sigma_denoiser: float = 0.05,
    regularizer: Optional[Regularizer] = None,
    rho: float = 1.0,
    max_iter: int = 30,
    tol: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Plug-and-Play ADMM denoiser.

    Args:
        noisy:           Noisy magnitude image, shape (1, 1, H, W), values in [0, 1].
        denoiser:        Denoising network. Must accept a (1, 2, H, W) tensor
                         [image_channel, sigma_map_channel] and return (1, 1, H, W).
        sigma_denoiser:  Noise level hint passed to the denoiser as a constant map.
        regularizer:     Proximal regularizer.  Defaults to ElasticNetRegularizer.
        rho:             ADMM penalty / step-size parameter.
        max_iter:        Maximum number of ADMM iterations.
        tol:             Relative change tolerance for early stopping.
        device:          Torch device.

    Returns:
        Denoised image, shape (1, 1, H, W), values clamped to [0, 1].
    """
    if regularizer is None:
        regularizer = ElasticNetRegularizer(lam1=0.01, lam2=0.01)

    noisy = noisy.to(device)
    denoiser = denoiser.to(device)
    denoiser.eval()

    # Constant sigma map broadcast to the same spatial size as noisy
    sigma_map = torch.full_like(noisy, sigma_denoiser)  # (1, 1, H, W)

    # ADMM variable initialisation
    x = noisy.clone()
    z = x.clone()
    u = torch.zeros_like(x)

    for _ in range(max_iter):
        x_prev = x.clone()

        # ------------------------------------------------------------------
        # x-update: apply denoiser to (z - u)
        # ------------------------------------------------------------------
        inp = torch.cat([z - u, sigma_map], dim=1)  # (1, 2, H, W)
        with torch.no_grad():
            x = denoiser(inp)

        # ------------------------------------------------------------------
        # z-update: proximal operator of regularizer
        # ------------------------------------------------------------------
        z = regularizer.prox(x + u, rho)

        # ------------------------------------------------------------------
        # u-update: dual variable (scaled form)
        # ------------------------------------------------------------------
        u = u + x - z

        # ------------------------------------------------------------------
        # Convergence check
        # ------------------------------------------------------------------
        rel_change = (x - x_prev).norm() / (x_prev.norm() + 1e-8)
        if rel_change < tol:
            break

    return x.clamp(0.0, 1.0)
