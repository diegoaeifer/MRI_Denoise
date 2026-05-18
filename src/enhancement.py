"""Post-processing enhancement pipeline for denoised MRI slices.

Supported step types (configure via list of dicts):
  clahe        clip_limit=0.01, kernel_size=None, nbins=256
  unsharp_mask radius=1.0, amount=0.5
  dither       bits=7  (Bayer ordered; 2^bits quantization levels)
  interpolate  scale_factor=2.0, order=3 (bicubic)

All steps take float32 [0,1] and return float32 [0,1].

Example
-------
>>> pipe = EnhancementPipeline([
...     {"type": "clahe", "clip_limit": 0.01},
...     {"type": "unsharp_mask", "radius": 1.0, "amount": 0.3},
... ])
>>> enhanced = pipe.apply(denoised_slice)
"""
from __future__ import annotations
from typing import Any
import numpy as np

_VALID_STEPS = {"clahe", "unsharp_mask", "dither", "interpolate"}


class EnhancementPipeline:
    def __init__(self, steps: list[dict[str, Any]]) -> None:
        for s in steps:
            if s.get("type") not in _VALID_STEPS:
                raise ValueError(
                    f"Unknown enhancement step: {s.get('type')!r}. "
                    f"Valid: {sorted(_VALID_STEPS)}"
                )
        self.steps = steps

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply all steps sequentially. Input/output: float32 (H,W) in [0,1]."""
        img = img.astype(np.float32)
        for step in self.steps:
            t = step["type"]
            kw = {k: v for k, v in step.items() if k != "type"}
            if t == "clahe":
                img = _clahe(img, **kw)
            elif t == "unsharp_mask":
                img = _unsharp(img, **kw)
            elif t == "dither":
                img = _dither(img, **kw)
            elif t == "interpolate":
                img = _interpolate(img, **kw)
        return img


def _clahe(img: np.ndarray,
            clip_limit: float = 0.01,
            kernel_size: int | None = None,
            nbins: int = 256) -> np.ndarray:
    from skimage.exposure import equalize_adapthist
    return np.clip(
        equalize_adapthist(img, kernel_size=kernel_size,
                           clip_limit=clip_limit, nbins=nbins),
        0.0, 1.0
    ).astype(np.float32)


def _unsharp(img: np.ndarray,
              radius: float = 1.0,
              amount: float = 0.5) -> np.ndarray:
    from skimage.filters import unsharp_mask
    return np.clip(
        unsharp_mask(img, radius=radius, amount=amount,
                     preserve_range=True, channel_axis=None),
        0.0, 1.0
    ).astype(np.float32)


def _dither(img: np.ndarray, bits: int = 7) -> np.ndarray:
    """Bayer 4x4 ordered dithering quantized to 2^bits levels."""
    levels = 2 ** bits
    bayer4 = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5],
    ], dtype=np.float32) / 16.0
    H, W = img.shape
    threshold = np.tile(bayer4, ((H + 3) // 4, (W + 3) // 4))[:H, :W]
    dithered = np.floor(img * (levels - 1) + threshold)
    return (np.clip(dithered, 0, levels - 1) / (levels - 1)).astype(np.float32)


def _interpolate(img: np.ndarray,
                 scale_factor: float = 2.0,
                 order: int = 3) -> np.ndarray:
    from scipy.ndimage import zoom
    return np.clip(
        zoom(img, scale_factor, order=order, prefilter=True),
        0.0, 1.0
    ).astype(np.float32)
