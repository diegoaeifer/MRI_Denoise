"""
Zero-shot benchmark of all pretrained denoising weights.

Loads DICOM test volumes, adds synthetic noise (Rician + acceleration artifacts),
runs every registered model, computes PSNR/SSIM/HaarPSI/VGG/sharpness,
and writes a CSV leaderboard + PNG montage.

Usage
-----
    python scripts/benchmark_pretrained.py \\
        --data C:/projetos/Datasets/dcm_BB_noisy \\
        --output C:/projetos/MRI_Denoise/artifacts/benchmark \\
        --noise-levels 0.02 0.05 0.10 \\
        --models all

    # Single model, specific noise:
    python scripts/benchmark_pretrained.py --models drunet gsdrunet --noise-levels 0.05
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# --- path setup so we can import from src/ when run from project root ---
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT.parent / "mr4raw" / "src"))
sys.path.insert(0, str(_ROOT.parent / "WCRR" / "wcrr-mri-pipeline" / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse)


def ssim_2d(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity
        p = pred.squeeze().cpu().numpy()
        t = target.squeeze().cpu().numpy()
        return structural_similarity(p, t, data_range=1.0)
    except Exception:
        return float("nan")


def haarpsi(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from wcrr_mri.metrics.haarpsi import HaarPSI
        score = HaarPSI()(pred.unsqueeze(0), target.unsqueeze(0))
        return score.item()
    except Exception:
        return float("nan")


def vgg_sim(pred: torch.Tensor, target: torch.Tensor, device: torch.device) -> float:
    try:
        from losses.auxiliary import VGGPerceptualLoss
        vgg = VGGPerceptualLoss().to(device)
        # VGG expects 3-channel; repeat grayscale
        p3 = pred.repeat(1, 3, 1, 1) if pred.shape[1] == 1 else pred
        t3 = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        loss = vgg(p3.to(device), t3.to(device)).item()
        return loss  # lower = better
    except Exception:
        return float("nan")


def sharpness(img: torch.Tensor) -> float:
    """Laplacian variance — higher = sharper."""
    arr = img.squeeze().cpu().numpy().astype(np.float32)
    from scipy.ndimage import laplace
    lap = laplace(arr)
    return float(lap.var())


# ---------------------------------------------------------------------------
# Noise synthesis
# ---------------------------------------------------------------------------

def add_rician_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Rician noise: sqrt((x + n1)^2 + n2^2) where n1,n2 ~ N(0,sigma)."""
    n1 = np.random.normal(0, sigma, img.shape).astype(np.float32)
    n2 = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.sqrt((img + n1) ** 2 + n2 ** 2)


def make_sigma_map(shape: tuple, sigma: float, device: torch.device) -> torch.Tensor:
    return torch.full((1, 1, shape[-2], shape[-1]), sigma, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dicom_slices(dicom_dir: Path, max_slices: int = 10) -> list[np.ndarray]:
    """Load up to max_slices 2D slices from a DICOM directory, normalised [0,1]."""
    import pydicom
    files = sorted(dicom_dir.rglob("*.dcm"))[:max_slices]
    if not files:
        # Try without extension (some DICOMs have none)
        files = [f for f in sorted(dicom_dir.rglob("*")) if f.is_file()][:max_slices]
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
            arr = ds.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            slices.append(arr)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    return slices


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _default_config() -> dict:
    """Minimal config that satisfies the factory for all pretrained models."""
    return {
        "common": {"in_channels": 2, "out_channels": 1},
        "drunet": {"base_channels": 64},
        "nafnet": {"width": 32, "enc_blk_nums": [1, 1, 1, 28], "middle_blk_num": 1, "dec_blk_nums": [1, 1, 1, 1]},
        "nafnet_xs": {"width": 16, "enc_blk_nums": [1, 1, 1, 1], "middle_blk_num": 1, "dec_blk_nums": [1, 1, 1, 1]},
        "scunet": {"config": [4, 4, 4, 4, 4, 4, 4]},
        "unet": {"bilinear": True},
        "ffdnet": {"nc": 64, "nb": 15},
        "se_scunet_mini": {"config": [1, 1, 1, 1, 1, 1, 1], "dim": 64},
        "snraware": {
            "model_path": str(_ROOT / "weights" / "SNRAware" / "small" / "snraware_small_model.pts"),
            "overlap": 16,
            "freeze": True,
        },
        "gsdrunet": {"pretrained": "download"},
        "nafnet_small": {"width": 32, "enc_blk_nums": [1, 1, 1, 4], "middle_blk_num": 1, "dec_blk_nums": [1, 1, 1, 1]},
        "imt_mrd": {"model_path": None, "freeze_backbone": True},
        "cdlnet": {"K": 3, "M": 64, "P": 7, "s": 1, "adaptive": False, "init": False},
        "restore_rwkv": {"dim": 48, "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4},
        "astro_denoiser": {"filters": 32, "depth": 6},
    }


# Models to benchmark in order (skip if weight load fails)
ALL_MODELS = [
    "drunet_pretrained",
    "gsdrunet",
    "restormer",
    "swinir_pretrained",
    "dncnn_pretrained",
    "scunet_pretrained",
    "nafnet_xs",
    "nafnet_small",
    "unet",
    "ffdnet",
    "visnet",
    "snraware",
    "imt-mrd",
    "astro_denoiser",
    "cdlnet",
    "restore_rwkv",
    "bm3d",
    "ram_pretrained",
]


def load_model(name: str, device: torch.device) -> Optional[torch.nn.Module]:
    try:
        from models import get_model
        model = get_model(name, _default_config())
        model.eval().to(device)
        log.info(f"  Loaded: {name}")
        return model
    except Exception as e:
        log.warning(f"  SKIP {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_model(model: torch.nn.Module, noisy: np.ndarray, sigma: float,
              device: torch.device) -> np.ndarray:
    t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    s = make_sigma_map(t.shape, sigma, device)
    inp = torch.cat([t, s], dim=1)  # (1,2,H,W)
    out = model(inp)
    return out.squeeze().cpu().numpy()


def benchmark(
    data_dir: Path,
    output_dir: Path,
    noise_levels: list[float],
    model_names: list[str],
    max_slices: int = 10,
    device: torch.device = torch.device("cpu"),
):
    import pandas as pd
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading DICOM slices from {data_dir}")
    slices = load_dicom_slices(data_dir, max_slices)
    if not slices:
        raise RuntimeError(f"No DICOM slices found in {data_dir}")
    log.info(f"Loaded {len(slices)} slices")

    log.info("Loading models...")
    models = {}
    for name in model_names:
        m = load_model(name, device)
        if m is not None:
            models[name] = m

    records = []
    for sigma in noise_levels:
        log.info(f"\n=== Noise level σ={sigma:.3f} ===")
        for i, clean in enumerate(slices):
            noisy = add_rician_noise(clean, sigma)
            noisy_clipped = np.clip(noisy, 0, 1)

            # Noisy baseline metrics
            records.append({
                "model": "noisy_input",
                "sigma": sigma,
                "slice": i,
                "psnr": psnr(torch.tensor(noisy_clipped).unsqueeze(0).unsqueeze(0),
                             torch.tensor(clean).unsqueeze(0).unsqueeze(0)),
                "ssim": ssim_2d(torch.tensor(noisy_clipped), torch.tensor(clean)),
                "haarpsi": float("nan"),
                "vgg_loss": float("nan"),
                "sharpness": sharpness(torch.tensor(noisy_clipped)),
            })

            for name, model in models.items():
                try:
                    denoised = run_model(model, noisy_clipped, sigma, device)
                    denoised = np.clip(denoised, 0, 1)
                    d_t = torch.tensor(denoised).unsqueeze(0).unsqueeze(0)
                    c_t = torch.tensor(clean).unsqueeze(0).unsqueeze(0)
                    records.append({
                        "model": name,
                        "sigma": sigma,
                        "slice": i,
                        "psnr": psnr(d_t, c_t),
                        "ssim": ssim_2d(d_t.squeeze(), c_t.squeeze()),
                        "haarpsi": haarpsi(d_t.squeeze(0), c_t.squeeze(0)),
                        "vgg_loss": vgg_sim(d_t, c_t, device),
                        "sharpness": sharpness(d_t),
                    })
                except Exception as e:
                    log.warning(f"  {name} failed on slice {i}: {e}")

    df = pd.DataFrame(records)

    # Save CSV
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nSaved: {csv_path}")

    # Leaderboard: mean across slices/sigmas
    leaderboard = (
        df[df["model"] != "noisy_input"]
        .groupby("model")[["psnr", "ssim", "sharpness"]]
        .mean()
        .sort_values("psnr", ascending=False)
    )
    log.info("\n=== LEADERBOARD (mean PSNR) ===\n" + leaderboard.to_string())

    lb_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(lb_path)

    _save_montage(slices, models, noise_levels[0], device, output_dir)

    return df


def _save_montage(slices, models, sigma, device, output_dir):
    """Save a visual montage of clean / noisy / denoised for the first slice."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clean = slices[0]
        noisy = np.clip(add_rician_noise(clean, sigma), 0, 1)

        cols = ["clean", "noisy"] + list(models.keys())
        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
        images = [clean, noisy]
        for model in models.values():
            try:
                images.append(np.clip(run_model(model, noisy, sigma, device), 0, 1))
            except Exception:
                images.append(np.zeros_like(clean))

        for ax, img, title in zip(axes, images, cols):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        fig.suptitle(f"σ={sigma:.3f} — zero-shot benchmark", fontsize=10)
        plt.tight_layout()
        path = output_dir / f"montage_sigma{sigma:.3f}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved montage: {path}")
    except Exception as e:
        log.warning(f"Montage failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot denoising benchmark")
    p.add_argument("--data", type=Path,
                   default=Path("C:/projetos/Datasets/dcm_BB_noisy"),
                   help="Directory containing test DICOM files")
    p.add_argument("--output", type=Path,
                   default=Path("C:/projetos/MRI_Denoise/artifacts/benchmark"),
                   help="Output directory for results")
    p.add_argument("--noise-levels", nargs="+", type=float,
                   default=[0.02, 0.05, 0.10],
                   help="Rician noise σ as fraction of max signal")
    p.add_argument("--models", nargs="+", default=["all"],
                   help=f"Models to benchmark. 'all' runs all. Options: {ALL_MODELS}")
    p.add_argument("--max-slices", type=int, default=10,
                   help="Max DICOM slices to load per dataset")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    log.info(f"Device: {device}")

    model_names = ALL_MODELS if "all" in args.models else args.models

    benchmark(
        data_dir=args.data,
        output_dir=args.output,
        noise_levels=args.noise_levels,
        model_names=model_names,
        max_slices=args.max_slices,
        device=device,
    )


if __name__ == "__main__":
    main()
