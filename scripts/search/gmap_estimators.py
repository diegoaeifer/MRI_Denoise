"""Pure numpy functions that estimate a g-factor/noise map from a magnitude image.

Output contract for all functions:
  - input:  np.ndarray, shape (H, W), dtype float32, range [0, 1]
  - output: np.ndarray, shape (H, W), dtype float32, clipped to [0.1, 10.0]
            values are *relative* — mean≈1 means "noise is uniform here"
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def gmap_uniform(mag: np.ndarray) -> np.ndarray:
    """All-ones map — SNRAware default baseline."""
    return np.ones(mag.shape, dtype=np.float32)


def gmap_local_variance(mag: np.ndarray, window: int = 7) -> np.ndarray:
    """Estimate local σ from rolling-window variance of the magnitude image."""
    mag_f = mag.astype(np.float64)
    mean = ndimage.uniform_filter(mag_f, size=window)
    mean_sq = ndimage.uniform_filter(mag_f ** 2, size=window)
    var = np.clip(mean_sq - mean ** 2, 0.0, None)
    std = np.sqrt(var + 1e-8)
    g = std / (std.mean() + 1e-8)
    return g.clip(0.1, 10.0).astype(np.float32)


def gmap_wavelet(mag: np.ndarray) -> np.ndarray:
    """High-frequency noise proxy from the DWT HH detail subband."""
    import pywt
    _, (LH, HL, HH) = pywt.dwt2(mag.astype(np.float64), "db1")
    hh_abs = np.abs(HH)
    hh_up = np.kron(hh_abs, np.ones((2, 2), dtype=np.float64))
    hh_up = hh_up[: mag.shape[0], : mag.shape[1]]
    g = hh_up / (hh_up.mean() + 1e-8)
    return g.clip(0.1, 10.0).astype(np.float32)


def gmap_gradient(mag: np.ndarray) -> np.ndarray:
    """Inverted gradient magnitude — smooth regions get higher gmap (more noise)."""
    gx = ndimage.sobel(mag.astype(np.float64), axis=0)
    gy = ndimage.sobel(mag.astype(np.float64), axis=1)
    gmag = np.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    g_inv = 1.0 / (gmag / (gmag.mean() + 1e-8) + 0.1)
    g_inv = g_inv / (g_inv.mean() + 1e-8)
    return g_inv.clip(0.1, 10.0).astype(np.float32)


def gmap_mad(mag: np.ndarray, patch_size: int = 8) -> np.ndarray:
    """Patch-wise MAD estimator of local σ: σ̂ = MAD / 0.6745."""
    H, W = mag.shape
    out = np.empty_like(mag, dtype=np.float32)
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = mag[i : i + patch_size, j : j + patch_size]
            mad = float(np.median(np.abs(patch - np.median(patch))))
            sigma_hat = mad / 0.6745
            out[i : i + patch_size, j : j + patch_size] = sigma_hat
    g = out / (out.mean() + 1e-8)
    return g.clip(0.1, 10.0).astype(np.float32)


GMAP_FNS: dict[str, object] = {
    "uniform":     gmap_uniform,
    "local_var_5": lambda m: gmap_local_variance(m, window=5),
    "local_var_9": lambda m: gmap_local_variance(m, window=9),
    "wavelet":     gmap_wavelet,
    "gradient":    gmap_gradient,
    "mad_8":       lambda m: gmap_mad(m, patch_size=8),
}
