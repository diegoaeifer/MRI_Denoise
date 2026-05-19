"""Dispatch unsharp filter configs to the unsharp-mask package.

Adds C:/projetos/unsharp-mask to sys.path so filters.py and config.py are importable.
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

# postprocess.py is at MRI_Denoise/scripts/search/postprocess.py
# parents[0] = scripts/search
# parents[1] = scripts
# parents[2] = MRI_Denoise
# parents[3] = C:\projetos
_UNSHARP_DIR = Path(__file__).resolve().parents[3] / "unsharp-mask"
if str(_UNSHARP_DIR) not in sys.path:
    sys.path.insert(0, str(_UNSHARP_DIR))

if not _UNSHARP_DIR.exists():
    import warnings
    warnings.warn(
        f"unsharp-mask package not found at {_UNSHARP_DIR}; "
        "apply_unsharp will fail for non-'none' configs.",
        stacklevel=1,
    )


def apply_unsharp(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply a post-denoise sharpening filter.

    cfg shapes:
      {"name": "none"}
      {"name": "gsum", "intensity": 1.0}
      {"name": "unsharp", "amount": 2.0}
      {"name": "mlvum", "scale": 3.0}

    input/output: np.ndarray (H, W), float32, range [0, 1]
    """
    name = cfg["name"]
    if name == "none":
        return img.copy()

    from filters import gsum_approximation, simple_unsharp_mask, mlvum_filter  # type: ignore

    if name == "gsum":
        return gsum_approximation(img, intensity=cfg["intensity"]).astype(np.float32)
    if name == "unsharp":
        return simple_unsharp_mask(img, amount=cfg["amount"]).astype(np.float32)
    if name == "mlvum":
        return mlvum_filter(img, scale=cfg["scale"]).astype(np.float32)

    raise ValueError(f"Unknown unsharp filter: {name!r}")
