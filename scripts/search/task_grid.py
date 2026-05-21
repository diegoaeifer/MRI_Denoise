# MRI_Denoise/scripts/search/task_grid.py
"""Builds the full Cartesian task grid for the denoiser parameter search.

build_task_grid(seed) -> list[dict]  (shuffled, reproducible)
"""
from __future__ import annotations
import random
import itertools

MODELS_2D = [
    "drunet_pretrained",
    "restormer",
    "gsdrunet",
    "snraware_small",
    "snraware_medium",
    "snraware_large",
    "imt-mrd_complex",
    "imt-mrd_residual",
]

MODELS_3D = [
    "snraware_small",
    "snraware_medium",
    "snraware_large",
    "imt-mrd_complex",
    "imt-mrd_residual",
]

SIGMAS = [0.1, 0.2]

GMAP_STRATEGIES = [
    "uniform",
    "local_var_5",
    "local_var_9",
    "wavelet",
    "gradient",
    "mad_8",
]

UNSHARP_CFGS = [
    {"name": "none"},
    {"name": "gsum",    "intensity": 0.5},
    {"name": "gsum",    "intensity": 1.0},
    {"name": "gsum",    "intensity": 1.5},
    {"name": "unsharp", "amount": 1.0},
    {"name": "unsharp", "amount": 2.0},
    {"name": "unsharp", "amount": 3.0},
    {"name": "mlvum",   "scale": 1.0},
    {"name": "mlvum",   "scale": 3.0},
    {"name": "mlvum",   "scale": 5.0},
]

DITHER_CFGS = [
    {"name": "none"},
    {"name": "gaussian", "strength": 0.005},
    {"name": "gaussian", "strength": 0.01},
    {"name": "blue",     "strength": 0.005},
    {"name": "blue",     "strength": 0.01},
]


def build_task_grid(seed: int = 0) -> list[dict]:
    """Return a shuffled list of all (model, sigma, gmap, unsharp, dither, mode) combos."""
    tasks: list[dict] = []

    # 2D tasks — all models
    for model, sigma, gmap, unsharp, dither in itertools.product(
        MODELS_2D, SIGMAS, GMAP_STRATEGIES, UNSHARP_CFGS, DITHER_CFGS
    ):
        tasks.append({
            "model_name":    model,
            "sigma":         sigma,
            "gmap_strategy": gmap,
            "unsharp_cfg":   unsharp,
            "dither_cfg":    dither,
            "mode":          "2d",
        })

    # 3D tasks — only models that support 3D
    for model, sigma, gmap, unsharp, dither in itertools.product(
        MODELS_3D, SIGMAS, GMAP_STRATEGIES, UNSHARP_CFGS, DITHER_CFGS
    ):
        tasks.append({
            "model_name":    model,
            "sigma":         sigma,
            "gmap_strategy": gmap,
            "unsharp_cfg":   unsharp,
            "dither_cfg":    dither,
            "mode":          "3d",
        })

    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks
