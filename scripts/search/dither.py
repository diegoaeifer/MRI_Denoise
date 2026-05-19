"""Noise re-injection strategies applied AFTER denoising to recover micro-texture.

All functions:
  input:  np.ndarray (H, W) or (H, W, D), float32, range ≈ [0, 1]
  output: np.ndarray same shape, clipped to [0, 1], float32
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

_RNG = np.random.default_rng(1337)


def _dither_none(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    return img


def _dither_gaussian(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    noise = _RNG.standard_normal(img.shape).astype(np.float32) * sigma * strength
    return np.clip(img + noise, 0.0, 1.0)


def _dither_blue(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """High-pass filtered white noise ≈ blue noise (high frequency emphasis)."""
    white = _RNG.standard_normal(img.shape).astype(np.float32)
    if img.ndim == 2:
        low = ndimage.gaussian_filter(white, sigma=2.0)
    else:
        low = ndimage.gaussian_filter(white, sigma=(2.0, 2.0, 1.0))
    blue = white - low
    std = blue.std() + 1e-8
    return np.clip(img + blue / std * sigma * strength, 0.0, 1.0)


DITHER_FNS: dict[str, object] = {
    "none":     _dither_none,
    "gaussian": _dither_gaussian,
    "blue":     _dither_blue,
}


def apply_dither(img: np.ndarray, sigma: float, cfg: dict) -> np.ndarray:
    """Dispatch a dither config dict to the right function.

    cfg: {"name": "none"} | {"name": "gaussian", "strength": 0.01} | {"name": "blue", "strength": 0.01}
    """
    name = cfg["name"]
    strength = cfg.get("strength", 0.0)
    fn = DITHER_FNS[name]
    return fn(img, sigma, strength)
