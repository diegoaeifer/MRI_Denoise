"""Evaluate NLmCED on 1 IXI volume per sequence type at a given Rician sigma.

Usage:
    python MRI_Denoise/scripts/eval_nlmced_ixi.py [--sigma 0.1] [--iters 1]
        [--rho 0.01] [--alpha 0.01] [--num 1]

Logs per-volume slice-averaged metrics (PSNR, SSIM, NRMSE, HaarPSI) to stdout
and appends a row to benchmark_results/nlmced_ixi_eval.csv.

Author reference defaults: iter=1, rho=0.01, alpha=0.01, num=1.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MRI_Denoise.src.data.benchmark_loaders import IXILoader
from MRI_Denoise.src.models.nlmced_wrapper import nlmced_2d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
except ImportError:
    log.error("scikit-image not found — pip install scikit-image")
    sys.exit(1)

try:
    from piq import haarpsi as _haarpsi
    _HAS_HAARPSI = True
except ImportError:
    log.warning("piq not found — HaarPSI will be NaN (pip install piq)")
    _HAS_HAARPSI = False


# --------------------------------------------------------------------------- #
# noise & metric helpers
# --------------------------------------------------------------------------- #

RNG = np.random.default_rng(42)


def add_rician(img: np.ndarray, sigma: float) -> np.ndarray:
    n1 = RNG.normal(0, sigma, img.shape).astype(np.float32)
    n2 = RNG.normal(0, sigma, img.shape).astype(np.float32)
    return np.sqrt((img + n1) ** 2 + n2 ** 2).clip(0, 1)


def psnr(clean: np.ndarray, pred: np.ndarray) -> float:
    return float(_psnr(clean, pred, data_range=1.0))


def ssim(clean: np.ndarray, pred: np.ndarray) -> float:
    return float(_ssim(clean, pred, data_range=1.0))


def nrmse(clean: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sqrt(np.mean(clean ** 2))
    return float(np.sqrt(np.mean((clean - pred) ** 2)) / max(denom, 1e-8))


def haarpsi(clean: np.ndarray, pred: np.ndarray) -> float:
    """HaarPSI via piq — higher is better, range ≈ [0, 1]."""
    if not _HAS_HAARPSI:
        return float("nan")
    try:
        # piq expects (B, C, H, W) float32 tensors in [0, 1]
        c = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        p = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return float(_haarpsi(p, c, data_range=1.0).item())
    except Exception as e:
        log.debug(f"HaarPSI failed: {e}")
        return float("nan")


# --------------------------------------------------------------------------- #
# per-volume evaluation
# --------------------------------------------------------------------------- #

def eval_volume(
    vol: np.ndarray,
    sigma: float,
    iterations: int,
    rho: float,
    alpha: float,
    num: int,
    skip_empty: float = 0.02,
) -> dict:
    """Evaluate NLmCED slice-by-slice on a (H, W, D) volume.

    Skips background slices (max intensity < skip_empty).
    Returns dict of slice-averaged metrics + timing.
    """
    _, _, D = vol.shape
    noisy_vol = add_rician(vol, sigma)

    metrics_noisy: dict[str, list] = {"psnr": [], "ssim": [], "nrmse": [], "haarpsi": []}
    metrics_out:   dict[str, list] = {"psnr": [], "ssim": [], "nrmse": [], "haarpsi": []}
    t0 = time.perf_counter()

    for d in range(D):
        sl_clean = vol[:, :, d].astype(np.float64)
        sl_noisy = noisy_vol[:, :, d].astype(np.float64)

        if sl_clean.max() < skip_empty:
            continue

        sl_den = nlmced_2d(sl_noisy, iterations=iterations, rho=rho,
                           alpha=alpha, num=num)

        metrics_noisy["psnr"].append(psnr(sl_clean, sl_noisy))
        metrics_noisy["ssim"].append(ssim(sl_clean, sl_noisy))
        metrics_noisy["nrmse"].append(nrmse(sl_clean, sl_noisy))
        metrics_noisy["haarpsi"].append(haarpsi(sl_clean, sl_noisy))

        metrics_out["psnr"].append(psnr(sl_clean, sl_den))
        metrics_out["ssim"].append(ssim(sl_clean, sl_den))
        metrics_out["nrmse"].append(nrmse(sl_clean, sl_den))
        metrics_out["haarpsi"].append(haarpsi(sl_clean, sl_den))

    elapsed = time.perf_counter() - t0
    n = max(len(metrics_out["psnr"]), 1)

    def avg(lst):
        valid = [v for v in lst if not np.isnan(v)]
        return round(float(np.mean(valid)) if valid else float("nan"), 4)

    return {
        "slices_evaluated": n,
        "time_s": round(elapsed, 2),
        "psnr_noisy":     avg(metrics_noisy["psnr"]),
        "ssim_noisy":     avg(metrics_noisy["ssim"]),
        "nrmse_noisy":    avg(metrics_noisy["nrmse"]),
        "haarpsi_noisy":  avg(metrics_noisy["haarpsi"]),
        "psnr":           avg(metrics_out["psnr"]),
        "ssim":           avg(metrics_out["ssim"]),
        "nrmse":          avg(metrics_out["nrmse"]),
        "haarpsi":        avg(metrics_out["haarpsi"]),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate NLmCED on IXI dataset")
    ap.add_argument("--sigma",  type=float, default=0.1)
    ap.add_argument("--iters",  type=int,   default=1)
    ap.add_argument("--rho",    type=float, default=0.01)
    ap.add_argument("--alpha",  type=float, default=0.01)
    ap.add_argument("--num",    type=int,   default=1)
    ap.add_argument("--csv",    type=str,   default="nlmced_ixi_eval.csv",
                    help="Output CSV filename (relative to benchmark_results/)")
    args = ap.parse_args()

    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / args.csv

    fieldnames = [
        "dataset", "modality", "subject_id", "method", "sigma",
        "iterations", "rho", "alpha", "num",
        "slices_evaluated",
        "psnr_noisy", "ssim_noisy", "nrmse_noisy", "haarpsi_noisy",
        "psnr", "ssim", "nrmse", "haarpsi",
        "time_s",
    ]

    write_header = not csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    log.info(
        f"NLmCED | sigma={args.sigma} | iters={args.iters} | "
        f"rho={args.rho} | alpha={args.alpha} | num={args.num}"
    )
    log.info("-" * 70)

    loader = IXILoader()
    for entry in loader.volumes():
        vol      = entry["volume"]
        modality = entry["modality"]
        subject  = entry["subject_id"]

        log.info(f"[{modality:>3}] {subject}  shape={vol.shape}  evaluating …")

        m = eval_volume(vol, args.sigma, iterations=args.iters,
                        rho=args.rho, alpha=args.alpha, num=args.num)

        writer.writerow({
            "dataset": "IXI", "modality": modality, "subject_id": subject,
            "method": "nlmced", "sigma": args.sigma,
            "iterations": args.iters, "rho": args.rho,
            "alpha": args.alpha, "num": args.num,
            **m,
        })
        csv_file.flush()

        log.info(
            f"[{modality:>3}] slices={m['slices_evaluated']}  "
            f"PSNR    noisy={m['psnr_noisy']:.2f}  → {m['psnr']:.2f} dB  "
            f"(Δ{m['psnr'] - m['psnr_noisy']:+.2f})"
        )
        log.info(
            f"[{modality:>3}]          "
            f"SSIM    noisy={m['ssim_noisy']:.4f}  → {m['ssim']:.4f}  "
            f"(Δ{m['ssim'] - m['ssim_noisy']:+.4f})"
        )
        log.info(
            f"[{modality:>3}]          "
            f"HaarPSI noisy={m['haarpsi_noisy']:.4f}  → {m['haarpsi']:.4f}  "
            f"(Δ{m['haarpsi'] - m['haarpsi_noisy']:+.4f})"
        )
        log.info(
            f"[{modality:>3}]          "
            f"NRMSE   noisy={m['nrmse_noisy']:.4f}  → {m['nrmse']:.4f}  "
            f"time={m['time_s']:.1f}s"
        )
        log.info("-" * 70)

    csv_file.close()
    log.info(f"Results appended to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
