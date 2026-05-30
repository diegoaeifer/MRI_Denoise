"""NLmCED filter — Non-Local mean Coherence Enhancing Diffusion.

Python port of the MATLAB implementation by F. Romdhane et al.
Original MATLAB: https://github.com/FerielRamdhane/Denoising-CEST-MRI/tree/main/NLmCED_Filter
Docker/XNAT wrapper: https://github.com/FerielRamdhane/NLmCED-Filter
Reference: F. Romdhane, F. Benzarti, A. Hamid, IJCVR 2018 8:1.
DOI: 10.1504/IJCVR.2018.090012

Implements both 2D (slice-wise) and 3D (full-volume) variants.
NLmCEDWrapper dispatches automatically based on input tensor rank:
  (B, 2, H, W)    → 2D per-image
  (B, 2, H, W, D) → 3D per-volume

Author's reference defaults: iter=1, rho=0.01, alpha=0.01 (rho,alpha ∈ [0,0.1]).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False

# --------------------------------------------------------------------------- #
# Shared: Rician noise estimation
# --------------------------------------------------------------------------- #

def _rician_sigma(img: np.ndarray) -> float:
    """Wavelet MAD estimator for Rician noise std (works 2-D and 3-D)."""
    if _HAS_PYWT:
        if img.ndim == 2:
            _, (_, _, detail) = pywt.dwt2(img, "db1")
        else:
            coeffs = pywt.dwtn(img, "db1")
            key = "d" * img.ndim
            detail = coeffs.get(key, list(coeffs.values())[-1])
        sigma = float(np.median(np.abs(detail))) / 0.6745
    else:
        diff = img[1:] - img[:-1]
        sigma = float(np.median(np.abs(diff))) / 0.6745
    return max(sigma, 1e-8)


# =========================================================================== #
#  2-D IMPLEMENTATION
# =========================================================================== #

# Scharr rotation-invariant 2-D gradients (matches derivatives.m, m=1)
_SCHARR2D_ROW = np.array(
    [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float64
) / 32
_SCHARR2D_COL = np.array(
    [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float64
) / 32


def _scharr_gradients_2d(img: np.ndarray):
    from scipy.ndimage import convolve
    Grow = convolve(img, _SCHARR2D_ROW, mode="reflect")
    Gcol = convolve(img, _SCHARR2D_COL, mode="reflect")
    return Grow, Gcol


def _structure_tensor_2d(Gx, Gy, rho: float):
    if rho < 1e-6:
        return Gx * Gx, Gx * Gy, Gy * Gy
    return (
        gaussian_filter(Gx * Gx, sigma=rho),
        gaussian_filter(Gx * Gy, sigma=rho),
        gaussian_filter(Gy * Gy, sigma=rho),
    )


def _eigenvectors_2d(Jxx, Jxy, Jyy):
    """Closed-form 2×2 eigendecomposition, fully vectorised."""
    trace = Jxx + Jyy
    disc  = np.sqrt(np.maximum((Jxx - Jyy) ** 2 / 4 + Jxy ** 2, 0.0))
    mu1   = trace / 2 - disc   # smaller
    mu2   = trace / 2 + disc   # larger

    denom = np.sqrt((mu1 - Jxx) ** 2 + Jxy ** 2 + 1e-16)
    v1x   = Jxy / denom
    v1y   = (mu1 - Jxx) / denom
    v2x, v2y = -v1y, v1x      # perpendicular

    return mu1, mu2, v1x, v1y, v2x, v2y


def _diffusion_tensor_2d(mu1, mu2, v1x, v1y, v2x, v2y, alpha: float):
    """2-D CED diffusion tensor: λ1 along edge (v1), λ2 across (v2)."""
    K    = np.maximum((mu2 - mu1) ** 2, 1e-15)
    lam1 = np.full_like(mu1, alpha)
    lam2 = alpha + (1.0 - alpha) * np.exp(-0.6931 / K)
    flat = (np.abs(mu1) < 1e-15) | (np.abs(mu2) < 1e-15)
    lam1[flat] = lam2[flat] = 0.0

    Dxx = lam1 * v1x**2 + lam2 * v2x**2
    Dyy = lam1 * v1y**2 + lam2 * v2y**2
    Dxy = lam1 * v1x * v1y + lam2 * v2x * v2y
    return Dxx, Dyy, Dxy


def _ced_step_2d(u, Dxx, Dyy, Dxy):
    """9-point 2-D CED stencil — 2-D reduction of MATLAB NLmCED.m B-block."""
    H, W  = u.shape
    px    = np.minimum(np.arange(H) + 1, H - 1)
    nx    = np.maximum(np.arange(H) - 1, 0)
    py    = np.minimum(np.arange(W) + 1, W - 1)
    ny    = np.maximum(np.arange(W) - 1, 0)

    a, b, d = Dxx, Dyy, Dxy

    B1 = 0.5 * (-d[nx, :] - d[:, py])
    B2 = b[:, py] + b
    B3 = 0.5 * (d[px, :] + d[:, py])
    B4 = a[nx, :] + a
    B6 = a[px, :] + a
    B7 = 0.5 * (d[nx, :] + d[:, ny])
    B8 = b[:, ny] + b
    B9 = 0.5 * (-d[px, :] - d[:, ny])

    return (
        B1 * (u[np.ix_(nx, py)] - u)
        + B2 * (u[:, py] - u)
        + B3 * (u[np.ix_(px, py)] - u)
        + B4 * (u[nx, :] - u)
        + B6 * (u[px, :] - u)
        + B7 * (u[np.ix_(nx, ny)] - u)
        + B8 * (u[:, ny] - u)
        + B9 * (u[np.ix_(px, ny)] - u)
    )


def _local_mean_2d(u, num: int):
    r = np.arange(-num, num + 1, dtype=np.float64)
    g = np.exp(-(r**2) / max(2 * num**2, 1.0))
    g /= g.sum()
    from scipy.ndimage import convolve
    return convolve(u, np.outer(g, g), mode="reflect")


def _nlm_vote_2d(u, mu_map, I_field, h: float, search_radius: int = 3):
    """Vectorised NLM vote (2-D). search_radius controls the search window: 1→3×3, 2→5×5, 3→7×7."""
    H, W  = u.shape
    rows  = np.arange(H)
    cols  = np.arange(W)
    h2    = max(h * h, 1e-30)
    out   = np.zeros_like(u)
    norm  = np.zeros_like(u)

    for dy in range(-search_radius, search_radius + 1):
        yi = np.clip(rows + dy, 0, H - 1)
        for dx in range(-search_radius, search_radius + 1):
            xi      = np.clip(cols + dx, 0, W - 1)
            vals_s  = u[np.ix_(yi, xi)]
            mu_s    = mu_map[np.ix_(yi, xi)]
            I_s     = I_field[np.ix_(yi, xi)]
            dists   = ((mu_s - mu_map)**2 + (I_s - I_field)**2)**2 / h2
            wis     = np.exp(-dists)
            out    += wis * vals_s
            norm   += wis

    return np.where(norm > 0, out / norm, u)


def nlmced_2d(
    img: np.ndarray,
    iterations: int = 2,
    rho: float = 0.01,
    alpha: float = 0.01,
    num: int = 1,
    search_radius: int = 3,
) -> np.ndarray:
    """Apply NLmCED to a single 2-D float64 image in [0, 1].

    Parameters
    ----------
    img           : (H, W) float64
    iterations    : filter passes (minimum 2 recommended)
    rho           : structure-tensor Gaussian std (author range [0, 0.1])
    alpha         : minimum diffusivity along edge tangent
    num           : local-mean Gaussian window half-size
    search_radius : NLM search window half-size (1→3×3, 2→5×5, 3→7×7)
    """
    u = img.astype(np.float64)
    for _ in range(max(iterations, 1)):
        sig     = _rician_sigma(u)
        h       = 0.08 * sig
        usigma  = gaussian_filter(u, sigma=max(sig, 0.5))
        Gx, Gy  = _scharr_gradients_2d(usigma)
        Jxx, Jxy, Jyy       = _structure_tensor_2d(Gx, Gy, rho)
        mu1, mu2, v1x, v1y, v2x, v2y = _eigenvectors_2d(Jxx, Jxy, Jyy)
        Dxx, Dyy, Dxy       = _diffusion_tensor_2d(mu1, mu2, v1x, v1y, v2x, v2y, alpha)
        I_field = _ced_step_2d(u, Dxx, Dyy, Dxy)
        mu_map  = _local_mean_2d(u, num)
        u       = _nlm_vote_2d(u, mu_map, I_field, h, search_radius=search_radius)
    return u


# =========================================================================== #
#  3-D IMPLEMENTATION
# =========================================================================== #

# 3-D Scharr kernels — direct translation of derivatives.m (m=1, 3D)
# Array layout: (row/H, col/W, depth/D) matching numpy (H,W,D) convention.
# MATLAB Stencil(:,:,z) → numpy kernel[:, :, z]

def _make_scharr3d():
    sy = np.zeros((3, 3, 3), dtype=np.float64)  # row (H) gradient
    sy[:, :, 0] = [[9, 30, 9], [0, 0, 0], [-9, -30, -9]]
    sy[:, :, 1] = [[30, 100, 30], [0, 0, 0], [-30, -100, -30]]
    sy[:, :, 2] = [[9, 30, 9], [0, 0, 0], [-9, -30, -9]]

    sx = np.zeros((3, 3, 3), dtype=np.float64)  # col (W) gradient
    sx[:, :, 0] = [[9, 0, -9], [30, 0, -30], [9, 0, -9]]
    sx[:, :, 1] = [[30, 0, -30], [100, 0, -100], [30, 0, -30]]
    sx[:, :, 2] = [[9, 0, -9], [30, 0, -30], [9, 0, -9]]

    sz = np.zeros((3, 3, 3), dtype=np.float64)  # depth (D) gradient
    sz[:, :, 0] = [[9, 30, 9], [30, 100, 30], [9, 30, 9]]
    sz[:, :, 1] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    sz[:, :, 2] = [[-9, -30, -9], [-30, -100, -30], [-9, -30, -9]]

    return sy / 512, sx / 512, sz / 512

_SCHARR3D_ROW, _SCHARR3D_COL, _SCHARR3D_DEP = _make_scharr3d()


def _scharr_gradients_3d(img: np.ndarray):
    """3-D Scharr gradients matching MATLAB derivatives.m (Gx='x', Gy='y', Gz='z')."""
    from scipy.ndimage import convolve
    Grow = convolve(img, _SCHARR3D_ROW, mode="reflect")
    Gcol = convolve(img, _SCHARR3D_COL, mode="reflect")
    Gdep = convolve(img, _SCHARR3D_DEP, mode="reflect")
    return Grow, Gcol, Gdep


def _structure_tensor_3d(Gx, Gy, Gz, rho: float):
    """Gaussian-smoothed 3-D outer-product structure tensor (6 components)."""
    def _smooth(arr):
        return arr if rho < 1e-6 else gaussian_filter(arr, sigma=rho)

    return (
        _smooth(Gx * Gx), _smooth(Gx * Gy), _smooth(Gx * Gz),
        _smooth(Gy * Gy), _smooth(Gy * Gz),
        _smooth(Gz * Gz),
    )


def _eigenvectors_3d(Jxx, Jxy, Jxz, Jyy, Jyz, Jzz):
    """Vectorised 3×3 symmetric eigendecomposition for every voxel.

    Uses np.linalg.eigh (LAPACK dsyevd) — eigenvalues sorted ascending:
    mu1 ≤ mu2 ≤ mu3.  Eigenvectors are columns of the returned matrix.
    Operates in float32 to halve peak memory on large volumes.
    """
    shape = Jxx.shape          # (H, W, D)
    N     = Jxx.size

    M = np.empty((N, 3, 3), dtype=np.float32)
    M[:, 0, 0] = Jxx.ravel().astype(np.float32)
    M[:, 1, 1] = Jyy.ravel().astype(np.float32)
    M[:, 2, 2] = Jzz.ravel().astype(np.float32)
    M[:, 0, 1] = M[:, 1, 0] = Jxy.ravel().astype(np.float32)
    M[:, 0, 2] = M[:, 2, 0] = Jxz.ravel().astype(np.float32)
    M[:, 1, 2] = M[:, 2, 1] = Jyz.ravel().astype(np.float32)

    vals, vecs = np.linalg.eigh(M)  # (N,3), (N,3,3) — ascending order

    # mu1=smallest, mu2=mid, mu3=largest (consistent with MATLAB EigenVectors3D)
    mu1 = vals[:, 0].reshape(shape)
    mu2 = vals[:, 1].reshape(shape)
    mu3 = vals[:, 2].reshape(shape)

    # vecs[:, component, eigenvalue_index]
    v1x = vecs[:, 0, 0].reshape(shape)
    v1y = vecs[:, 1, 0].reshape(shape)
    v1z = vecs[:, 2, 0].reshape(shape)
    v2x = vecs[:, 0, 1].reshape(shape)
    v2y = vecs[:, 1, 1].reshape(shape)
    v2z = vecs[:, 2, 1].reshape(shape)
    v3x = vecs[:, 0, 2].reshape(shape)
    v3y = vecs[:, 1, 2].reshape(shape)
    v3z = vecs[:, 2, 2].reshape(shape)

    return mu1, mu2, mu3, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z


def _diffusion_tensor_3d(
    mu1, mu2, mu3,
    v1x, v1y, v1z,
    v2x, v2y, v2z,
    v3x, v3y, v3z,
    alpha: float,
):
    """Full 3-D CED diffusion tensor (direct port of NLmCED.m eigenmode section)."""
    eps = 1e-15
    K   = np.maximum(
        (mu1 - mu2)**2 + (mu1 - mu3)**2 + (mu2 - mu3)**2, eps
    )
    Cp    = (mu1 - mu2) / (mu1 + mu3 + eps)
    Cl    = (mu2 - mu3) / (mu2 + mu3 + eps)
    Cedge = Cl * (1.0 - Cp)

    lam1  = np.full_like(mu1, alpha, dtype=np.float64)
    lam2  = np.abs(np.tanh(Cedge / K))
    lam3  = alpha + (1.0 - alpha) * np.exp(-0.6931 / K)

    flat = (np.abs(mu1) < eps) | (np.abs(mu2) < eps) | (np.abs(mu3) < eps)
    lam1[flat] = lam2[flat] = lam3[flat] = 0.0

    Dxx = lam1*v1x**2 + lam2*v2x**2 + lam3*v3x**2
    Dyy = lam1*v1y**2 + lam2*v2y**2 + lam3*v3y**2
    Dzz = lam1*v1z**2 + lam2*v2z**2 + lam3*v3z**2
    Dxy = lam1*v1x*v1y + lam2*v2x*v2y + lam3*v3x*v3y
    Dxz = lam1*v1x*v1z + lam2*v2x*v2z + lam3*v3x*v3z
    Dyz = lam1*v1y*v1z + lam2*v2y*v2z + lam3*v3y*v3z

    return Dxx, Dyy, Dzz, Dxy, Dxz, Dyz


def _ced_step_3d(u, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    """19-neighbor 3-D CED stencil — direct port of NLmCED.m A/B/C blocks."""
    H, W, D = u.shape
    rows = np.arange(H); cols = np.arange(W); deps = np.arange(D)

    px = np.minimum(rows + 1, H - 1);  nx = np.maximum(rows - 1, 0)
    py = np.minimum(cols + 1, W - 1);  ny = np.maximum(cols - 1, 0)
    pz = np.minimum(deps + 1, D - 1);  nz = np.maximum(deps - 1, 0)

    a, b, c, d, e, f = Dxx, Dyy, Dzz, Dxy, Dxz, Dyz

    # --- A block: z-1 layer ---
    A2 = 0.5 * (-f[:, :, nz] - f[:, py, :])
    A4 = 0.5 * ( e[:, :, nz] + e[nx, :, :])
    A5 =        c[:, :, nz] + c
    A6 = 0.5 * (-e[:, :, nz] - e[px, :, :])
    A8 = 0.5 * ( f[:, :, nz] + f[:, ny, :])

    # --- B block: same z layer ---
    B1 = 0.5 * (-d[nx, :, :] - d[:, py, :])
    B2 =        b[:, py, :] + b
    B3 = 0.5 * ( d[px, :, :] + d[:, py, :])
    B4 =        a[nx, :, :] + a
    B6 =        a[px, :, :] + a
    B7 = 0.5 * ( d[nx, :, :] + d[:, ny, :])
    B8 =        b[:, ny, :] + b
    B9 = 0.5 * (-d[px, :, :] - d[:, ny, :])

    # --- C block: z+1 layer ---
    C2 = 0.5 * ( f[:, :, pz] + f[:, py, :])
    C4 = 0.5 * (-e[:, :, pz] - e[nx, :, :])
    C5 =        c[:, :, pz] + c
    C6 = 0.5 * ( e[:, :, pz] + e[px, :, :])
    C8 = 0.5 * (-f[:, :, pz] - f[:, ny, :])

    return (
        # A neighbors (z-1)
        A2 * (u[np.ix_(rows, py,   nz)] - u)
      + A4 * (u[np.ix_(nx,   cols, nz)] - u)
      + A5 * (u[:, :, nz]               - u)
      + A6 * (u[np.ix_(px,   cols, nz)] - u)
      + A8 * (u[np.ix_(rows, ny,   nz)] - u)
        # B neighbors (same z)
      + B1 * (u[np.ix_(nx, py, deps)]   - u)
      + B2 * (u[:, py, :]               - u)
      + B3 * (u[np.ix_(px, py, deps)]   - u)
      + B4 * (u[nx, :, :]               - u)
      + B6 * (u[px, :, :]               - u)
      + B7 * (u[np.ix_(nx, ny, deps)]   - u)
      + B8 * (u[:, ny, :]               - u)
      + B9 * (u[np.ix_(px, ny, deps)]   - u)
        # C neighbors (z+1)
      + C2 * (u[np.ix_(rows, py,   pz)] - u)
      + C4 * (u[np.ix_(nx,   cols, pz)] - u)
      + C5 * (u[:, :, pz]               - u)
      + C6 * (u[np.ix_(px,   cols, pz)] - u)
      + C8 * (u[np.ix_(rows, ny,   pz)] - u)
    )


def _local_mean_3d(u, num: int):
    """3-D Gaussian local mean with effective half-size num."""
    sigma = max(num / 2.0, 0.5)
    return gaussian_filter(u.astype(np.float64), sigma=sigma)


def _nlm_vote_3d(u, mu_map, I_field, h: float, search_radius: int = 3):
    """Vectorised NLM vote (3-D). search_radius controls the search window: 1→3³, 2→5³, 3→7³."""
    H, W, D = u.shape
    rows = np.arange(H); cols = np.arange(W); deps = np.arange(D)
    h2   = max(h * h, 1e-30)
    out  = np.zeros_like(u)
    norm = np.zeros_like(u)

    for dz in range(-search_radius, search_radius + 1):
        zi = np.clip(deps + dz, 0, D - 1)
        for dy in range(-search_radius, search_radius + 1):
            yi = np.clip(rows + dy, 0, H - 1)
            for dx in range(-search_radius, search_radius + 1):
                xi     = np.clip(cols + dx, 0, W - 1)
                idx    = np.ix_(yi, xi, zi)
                vals_s = u[idx]
                mu_s   = mu_map[idx]
                I_s    = I_field[idx]
                dists  = ((mu_s - mu_map)**2 + (I_s - I_field)**2)**2 / h2
                wis    = np.exp(-dists)
                out   += wis * vals_s
                norm  += wis

    return np.where(norm > 0, out / norm, u)


def nlmced_3d(
    vol: np.ndarray,
    iterations: int = 2,
    rho: float = 0.01,
    alpha: float = 0.01,
    num: int = 1,
    search_radius: int = 3,
) -> np.ndarray:
    """Apply NLmCED to a 3-D volume (H, W, D) in [0, 1].

    Uses the full 3-D algorithm: 3-D Scharr gradients, 3×3 structure
    tensor eigendecomposition per voxel, 19-neighbor CED stencil, and
    NLM vote with configurable search window.

    Parameters
    ----------
    vol           : (H, W, D) float32 or float64
    iterations    : filter passes (minimum 2 recommended)
    rho           : structure-tensor Gaussian std (author range [0, 0.1])
    alpha         : minimum diffusivity
    num           : local-mean Gaussian window half-size
    search_radius : NLM search window half-size (1→3³, 2→5³, 3→7³)
    """
    u = vol.astype(np.float64)
    for _ in range(max(iterations, 1)):
        sig    = _rician_sigma(u)
        h      = 0.08 * sig
        usigma = gaussian_filter(u, sigma=max(sig, 0.5))

        Gx, Gy, Gz = _scharr_gradients_3d(usigma)

        Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = _structure_tensor_3d(Gx, Gy, Gz, rho)

        (mu1, mu2, mu3,
         v1x, v1y, v1z,
         v2x, v2y, v2z,
         v3x, v3y, v3z) = _eigenvectors_3d(Jxx, Jxy, Jxz, Jyy, Jyz, Jzz)

        Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = _diffusion_tensor_3d(
            mu1, mu2, mu3,
            v1x, v1y, v1z,
            v2x, v2y, v2z,
            v3x, v3y, v3z,
            alpha,
        )

        I_field = _ced_step_3d(u, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
        mu_map  = _local_mean_3d(u, num)
        u       = _nlm_vote_3d(u, mu_map, I_field, h, search_radius=search_radius)

    return u


# =========================================================================== #
#  WRAPPER — auto-dispatches 2-D vs 3-D
# =========================================================================== #

class NLmCEDWrapper(nn.Module):
    """Pipeline wrapper for the NLmCED filter.

    Input conventions
    -----------------
    2-D mode (B, 2, H, W)    → denoises each image independently → (B, 1, H, W)
    3-D mode (B, 2, H, W, D) → denoises each volume independently → (B, 1, H, W, D)

    ``mode='auto'`` selects based on input tensor rank (default).
    No learnable parameters — all computation is numpy/scipy on CPU.

    Parameters
    ----------
    mode       : 'auto' | '2d' | '3d'
    iterations : NLmCED passes (1 = author default)
    rho        : structure-tensor Gaussian std  (author range [0, 0.1])
    alpha      : minimum diffusivity along edge tangent
    num        : local-mean Gaussian window half-size
    """

    def __init__(
        self,
        mode: str = "auto",
        iterations: int = 2,
        rho: float = 0.01,
        alpha: float = 0.01,
        num: int = 1,
        search_radius: int = 3,
    ) -> None:
        super().__init__()
        if mode not in ("auto", "2d", "3d"):
            raise ValueError(f"mode must be 'auto', '2d', or '3d', got '{mode}'")
        self.mode          = mode
        self.iterations    = iterations
        self.rho           = rho
        self.alpha         = alpha
        self.num           = num
        self.search_radius = search_radius

    def _resolve_mode(self, x: torch.Tensor) -> str:
        if self.mode != "auto":
            return self.mode
        if x.ndim == 4:
            return "2d"
        if x.ndim == 5:
            return "3d"
        raise ValueError(f"Expected 4-D or 5-D input, got {x.ndim}-D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) → (B, 1, H, W)  or  (B, 2, H, W, D) → (B, 1, H, W, D)."""
        device  = x.device
        mode    = self._resolve_mode(x)
        B       = x.shape[0]
        kwargs  = dict(iterations=self.iterations, rho=self.rho,
                       alpha=self.alpha, num=self.num, search_radius=self.search_radius)
        outputs = []

        if mode == "2d":
            for b in range(B):
                img = x[b, 0].detach().cpu().numpy().astype(np.float64)
                outputs.append(torch.from_numpy(nlmced_2d(img, **kwargs)).float())
            return torch.stack(outputs).unsqueeze(1).to(device)

        else:  # 3d
            for b in range(B):
                vol = x[b, 0].detach().cpu().numpy().astype(np.float32)
                outputs.append(torch.from_numpy(nlmced_3d(vol, **kwargs)).float())
            return torch.stack(outputs).unsqueeze(1).to(device)
