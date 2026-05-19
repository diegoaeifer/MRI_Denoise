# MRI_Denoise/scripts/search/eval_pipeline.py
"""Single-trial runner for the denoiser parameter search.

run_trial(task: dict) -> dict
  task keys: model_name, sigma, gmap_strategy, unsharp_cfg, dither_cfg, mode
  result keys: psnr, ssim, elapsed_s, + all task keys, error (None on success)

Designed to run inside ProcessPoolExecutor workers (stateless per-call).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_MRI_ROOT    = Path(__file__).resolve().parents[2]
_PROJ_ROOT   = _MRI_ROOT.parent

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_MRI_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_MRI_ROOT / "src"))

from search.gmap_estimators import GMAP_FNS
from search.dither import apply_dither
from search.postprocess import apply_unsharp

IXI_T2    = _PROJ_ROOT / "Datasets" / "IXI" / "all" / "IXI015-HH-1258-T2.nii.gz"
IXI_PD    = _PROJ_ROOT / "Datasets" / "IXI" / "all" / "IXI015-HH-1258-PD.nii.gz"
SLICE_IDX = 60  # IXI T2 has 120 slices; 60 is the midpoint


def load_clean_2d() -> np.ndarray:
    if IXI_T2.exists():
        import nibabel as nib
        vol = nib.load(str(IXI_T2)).get_fdata(dtype=np.float32)
        img = vol[:, :, SLICE_IDX]
    else:
        import pydicom
        ds = pydicom.dcmread(pydicom.data.get_testdata_file("MR_small.dcm"))
        img = ds.pixel_array.astype(np.float32)
    return (img / (img.max() + 1e-8)).astype(np.float32)


def load_clean_3d() -> np.ndarray:
    if not IXI_PD.exists():
        raise RuntimeError(f"IXI PD not found at {IXI_PD}; 3D eval requires IXI dataset")
    import nibabel as nib
    vol = nib.load(str(IXI_PD)).get_fdata(dtype=np.float32)
    crop = vol[:, :, 80:144]  # 64-slice crop for speed
    return (crop / (crop.max() + 1e-8)).astype(np.float32)


def add_rician_noise(
    clean: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n1 = rng.standard_normal(clean.shape).astype(np.float32) * sigma
    n2 = rng.standard_normal(clean.shape).astype(np.float32) * sigma
    return np.sqrt((clean + n1) ** 2 + n2 ** 2).clip(0, None).astype(np.float32)


def build_input_tensor(
    noisy: np.ndarray,
    sigma: float,
    gmap_strategy: str,
    mode: str,
) -> torch.Tensor:
    """Build (B=1, C=2, ...) input tensor.

    2D: (1, 2, H, W)       ch0=noisy, ch1=gmap*sigma
    3D: (1, 2, D, H, W)    noisy is (H, W, D); output reordered to (D, H, W)
    """
    gmap_fn = GMAP_FNS[gmap_strategy]
    if mode == "2d":
        gmap = gmap_fn(noisy) * sigma  # (H, W)
        arr = np.stack([noisy, gmap], axis=0)[np.newaxis]  # (1, 2, H, W)
    else:
        H, W, D = noisy.shape
        gmap_slices = np.stack(
            [gmap_fn(noisy[:, :, d]) * sigma for d in range(D)], axis=2
        )  # (H, W, D)
        noisy_bhwd = noisy.transpose(2, 0, 1)[np.newaxis, np.newaxis]       # (1,1,D,H,W)
        gmap_bhwd  = gmap_slices.transpose(2, 0, 1)[np.newaxis, np.newaxis] # (1,1,D,H,W)
        arr = np.concatenate([noisy_bhwd, gmap_bhwd], axis=1)               # (1, 2, D, H, W)
    return torch.from_numpy(arr)


# ─── model loading ─────────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, torch.nn.Module] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_name: str, gmap_strategy: str) -> torch.nn.Module:
    cache_key = f"{model_name}|{gmap_strategy}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if model_name.startswith("snraware_"):
        size = model_name.split("_")[1]  # "small", "medium", "large"
        from models.snraware_wrapper import SNRAwareWrapper
        model = SNRAwareWrapper(model_size=size)
    elif model_name.startswith("imt-mrd_"):
        variant = model_name.split("_")[1]  # "complex", "residual"
        from models.imt_mrd_wrapper import ImtMrdWrapper
        model = ImtMrdWrapper(model_variant=variant, freeze_backbone=True)
    else:
        import deepinv as dinv
        _DI_MODELS = {
            "drunet_pretrained": lambda: dinv.models.DRUNet(
                in_channels=1, out_channels=1, pretrained="download"
            ),
            "dncnn_pretrained": lambda: dinv.models.DnCNN(
                in_channels=1, out_channels=1, pretrained="download"
            ),
            "swinir_pretrained": lambda: dinv.models.SwinIR(
                in_chans=1, pretrained="download"
            ),
            "restormer": lambda: dinv.models.Restormer(
                in_channels=1, out_channels=1, pretrained="denoising_gray"
            ),
            "gsdrunet": lambda: dinv.models.GSDRUNet(
                in_channels=1, pretrained="download"
            ),
        }
        if model_name not in _DI_MODELS:
            raise ValueError(f"Unknown model: {model_name!r}")
        model = _DI_MODELS[model_name]()

    model = model.to(DEVICE).eval()
    _MODEL_CACHE[cache_key] = model
    return model


# ─── forward pass helper ──────────────────────────────────────────────────

def _run_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma: float,
    model_name: str,
) -> np.ndarray:
    """Run forward pass, return denoised numpy in original spatial layout."""
    x = x.to(DEVICE)
    with torch.no_grad():
        if model_name.startswith(("snraware_", "imt-mrd_")):
            out = model(x)  # (B, 1, H, W) or (B, 1, D, H, W)
        else:
            img_1ch = x[:, 0:1]  # drop sigma channel
            sigma_t = torch.tensor([sigma], device=DEVICE)
            out = model(img_1ch, sigma_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.shape[1] == 3:
                out = out.mean(dim=1, keepdim=True)
    arr = out.cpu().squeeze().numpy()  # (H,W) or (D,H,W)
    # 3D TorchScript models output (D,H,W) — transpose back to (H,W,D)
    if arr.ndim == 3 and model_name.startswith(("snraware_", "imt-mrd_")):
        arr = arr.transpose(1, 2, 0)
    return arr.clip(0, 1).astype(np.float32)


# ─── main trial function ───────────────────────────────────────────────────

def run_trial(task: dict) -> dict:
    """Evaluate a full pipeline configuration.

    task keys:
      model_name    str   e.g. "drunet_pretrained"
      sigma         float e.g. 0.1
      gmap_strategy str   key in GMAP_FNS
      unsharp_cfg   dict  {"name": "gsum", "intensity": 1.0}
      dither_cfg    dict  {"name": "gaussian", "strength": 0.01}
      mode          str   "2d" or "3d"
    """
    t0 = time.monotonic()
    result = dict(task)
    result["error"] = None

    try:
        rng = np.random.default_rng(42)
        clean = load_clean_2d() if task["mode"] == "2d" else load_clean_3d()
        noisy = add_rician_noise(clean, task["sigma"], rng)
        x = build_input_tensor(noisy, task["sigma"], task["gmap_strategy"], task["mode"])

        model = _load_model(task["model_name"], task["gmap_strategy"])
        denoised = _run_model(model, x, task["sigma"], task["model_name"])

        if task["mode"] == "3d":
            H, W, D = denoised.shape
            pp = np.stack(
                [apply_unsharp(denoised[:, :, d], task["unsharp_cfg"]) for d in range(D)],
                axis=2,
            )
            final = np.stack(
                [apply_dither(pp[:, :, d], task["sigma"], task["dither_cfg"]) for d in range(D)],
                axis=2,
            )
        else:
            pp = apply_unsharp(denoised, task["unsharp_cfg"])
            final = apply_dither(pp, task["sigma"], task["dither_cfg"])

        result["psnr"] = float(
            peak_signal_noise_ratio(clean, final, data_range=1.0)
        )
        # channel_axis=2 treats D as channels — computes per-slice (H,W) SSIM and averages.
        # scikit-image has no native 3D SSIM; this is a consistent approximation.
        result["ssim"] = float(
            structural_similarity(
                clean, final, data_range=1.0,
                channel_axis=2 if task["mode"] == "3d" else None,
            )
        )
    except Exception as exc:
        result["psnr"] = None
        result["ssim"] = None
        result["error"] = str(exc)

    result["elapsed_s"] = time.monotonic() - t0
    return result


# ─── CPU stub for tests (no weights needed) ───────────────────────────────

def run_trial_cpu_stub(
    sigma: float,
    gmap_strategy: str,
    unsharp_cfg: dict,
    dither_cfg: dict,
) -> dict:
    """Lightweight version for tests: uses the noisy image itself as 'denoised'."""
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.2, 0.8, (64, 64)).astype(np.float32)
    noisy = add_rician_noise(clean, sigma, rng)
    x = build_input_tensor(noisy, sigma, gmap_strategy, "2d")

    denoised = x.squeeze()[0].numpy().clip(0, 1)  # ch0 = noisy image (identity stub)
    pp = apply_unsharp(denoised, unsharp_cfg)
    final = apply_dither(pp, sigma, dither_cfg)

    psnr = float(peak_signal_noise_ratio(clean, final, data_range=1.0))
    ssim = float(structural_similarity(clean, final, data_range=1.0))
    return {"psnr": psnr, "ssim": ssim, "error": None,
            "sigma": sigma, "gmap_strategy": gmap_strategy}
