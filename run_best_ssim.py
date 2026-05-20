"""
Run best-SSIM configs for GSD-DRUNet, WCRR-2D, WCRR-3D at sigma=0.1
on the T2 reference slice and save a comparison panel.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ── reuse all model/solver infrastructure from hparam_search ──────────────────
from hparam_search import (
    DEVICE, WCRR2D_MODEL, GSD_MODEL, WCRR3D_MODEL,
    np2t, t2np, nmapg, df_grad,
    run_wcrr2d, run_gsd, run_wcrr3d,
    load_ref_2d, load_ref_3d, add_rician,
)

SIGMA = 0.1
OUT_DIR = Path("C:/projetos/benchmark_results/best_ssim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best SSIM configs from best_configs.json
BEST_GSD = {
    "tau": 0.5, "n_iter": 80, "fidelity": "elasticnet",
    "alpha": 0.3, "beta_en": 0.3, "delta": 0.05,
}
BEST_WCRR2D = {
    "lam": 0.2, "step": 0.2, "n_iter": 300,
    "fidelity": "dynamic_elasticnet",
    "alpha": 0.5, "beta_en": 0.3, "delta": 0.05,
}
BEST_WCRR3D = {
    "lam": 0.15, "step": 0.1, "n_iter": 100,
    "fidelity": "dynamic_elasticnet",
    "alpha": 0.3, "beta_en": 0.3, "delta": 0.005,
}

def metrics(ref, rec):
    psnr = peak_signal_noise_ratio(ref, rec, data_range=1.0)
    ssim = structural_similarity(ref, rec, data_range=1.0)
    return psnr, ssim

def main():
    print(f"Device: {DEVICE}")
    print(f"sigma = {SIGMA}\n")

    ref_2d = load_ref_2d()
    ref_3d = load_ref_3d()

    np.random.seed(0)
    noisy_2d = add_rician(ref_2d, SIGMA)

    results = {}

    # ── GSD-DRUNet ────────────────────────────────────────────────────────────
    print("Running GSD-DRUNet (best SSIM)...")
    t0 = time.time()
    den_gsd = run_gsd(noisy_2d, SIGMA, BEST_GSD)
    elapsed = time.time() - t0
    p, s = metrics(ref_2d, den_gsd)
    print(f"  PSNR={p:.2f}  SSIM={s:.4f}  ({elapsed:.1f}s)")
    results["GSD-DRUNet"] = den_gsd

    # ── WCRR-2D ───────────────────────────────────────────────────────────────
    print("Running WCRR-2D (best SSIM)...")
    t0 = time.time()
    den_wcrr2d = run_wcrr2d(noisy_2d, BEST_WCRR2D)
    elapsed = time.time() - t0
    p, s = metrics(ref_2d, den_wcrr2d)
    print(f"  PSNR={p:.2f}  SSIM={s:.4f}  ({elapsed:.1f}s)")
    results["WCRR-2D"] = den_wcrr2d

    # ── WCRR-3D (middle slice of denoised volume) ─────────────────────────────
    print("Running WCRR-3D (best SSIM) — may take ~30-60s...")
    np.random.seed(0)
    noisy_3d = add_rician(ref_3d, SIGMA)
    t0 = time.time()
    den_vol = run_wcrr3d(noisy_3d, SIGMA, BEST_WCRR3D)
    elapsed = time.time() - t0
    mid = den_vol.shape[0] // 2
    den_wcrr3d_slice = den_vol[mid]
    ref_3d_slice = ref_3d[mid]
    noisy_3d_slice = noisy_3d[mid]
    p, s = metrics(ref_3d_slice, den_wcrr3d_slice)
    print(f"  PSNR={p:.2f}  SSIM={s:.4f}  ({elapsed:.1f}s)  [middle slice of 3D volume]")
    results["WCRR-3D (mid slice)"] = den_wcrr3d_slice

    # ── Save comparison panel (2D) ────────────────────────────────────────────
    n_cols = 2 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
    fig.suptitle(f"Best-SSIM Denoising  |  σ={SIGMA}  |  T2 reference", fontsize=13)

    def show(ax, img, title, ref=None):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        if ref is not None:
            p, s = metrics(ref, img)
            ax.set_title(f"{title}\nPSNR={p:.2f}  SSIM={s:.4f}", fontsize=8)
        else:
            ax.set_title(title, fontsize=9)
        ax.axis("off")

    show(axes[0], ref_2d,   "Reference (clean)")
    show(axes[1], noisy_2d, "Noisy",  ref=ref_2d)

    for i, (name, img) in enumerate(results.items()):
        ref_for_metric = ref_3d_slice if "3D" in name else ref_2d
        show(axes[2 + i], img, name, ref=ref_for_metric)

    plt.tight_layout()
    out_path = OUT_DIR / "best_ssim_panel.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPanel saved -> {out_path}")

    # ── Also save individual images ───────────────────────────────────────────
    def save_gray(arr, name):
        p = OUT_DIR / f"{name}.png"
        plt.imsave(str(p), arr, cmap="gray", vmin=0, vmax=1)
        return p

    save_gray(ref_2d,   "reference")
    save_gray(noisy_2d, "noisy")
    save_gray(den_gsd,  "gsd_best_ssim")
    save_gray(den_wcrr2d, "wcrr2d_best_ssim")
    save_gray(den_wcrr3d_slice, "wcrr3d_best_ssim_midslice")
    print("Individual PNGs saved to", OUT_DIR)

if __name__ == "__main__":
    main()
