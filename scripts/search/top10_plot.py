"""Run top-10 configurations at sigma=0.1 and sigma=0.2, then produce a large visual plot."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_MRI_ROOT    = Path(__file__).resolve().parents[2]
_PROJ_ROOT   = _MRI_ROOT.parent

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from search.eval_pipeline import (
    run_trial, load_clean_2d, load_clean_3d,
    add_rician_noise, build_input_tensor, _load_model, _run_model,
)
from search.postprocess import apply_unsharp
from search.dither import apply_dither
from search.gmap_estimators import GMAP_FNS

TOP10 = [
    {"model_name": "gsdrunet",          "gmap_strategy": "wavelet",    "unsharp_cfg": {"name": "mlvum",   "scale": 3.0}, "dither_cfg": {"name": "blue",     "strength": 0.005}, "mode": "2d"},
    {"model_name": "gsdrunet",          "gmap_strategy": "uniform",    "unsharp_cfg": {"name": "unsharp", "amount": 3.0}, "dither_cfg": {"name": "blue",     "strength": 0.005}, "mode": "2d"},
    {"model_name": "gsdrunet",          "gmap_strategy": "mad_8",      "unsharp_cfg": {"name": "unsharp", "amount": 3.0}, "dither_cfg": {"name": "gaussian", "strength": 0.005}, "mode": "2d"},
    {"model_name": "drunet_pretrained", "gmap_strategy": "wavelet",    "unsharp_cfg": {"name": "mlvum",   "scale": 3.0}, "dither_cfg": {"name": "blue",     "strength": 0.005}, "mode": "2d"},
    {"model_name": "drunet_pretrained", "gmap_strategy": "mad_8",      "unsharp_cfg": {"name": "mlvum",   "scale": 3.0}, "dither_cfg": {"name": "blue",     "strength": 0.005}, "mode": "2d"},
    {"model_name": "gsdrunet",          "gmap_strategy": "gradient",   "unsharp_cfg": {"name": "gsum",    "intensity": 1.5}, "dither_cfg": {"name": "blue", "strength": 0.005}, "mode": "2d"},
    {"model_name": "drunet_pretrained", "gmap_strategy": "local_var_9","unsharp_cfg": {"name": "gsum",    "intensity": 1.5}, "dither_cfg": {"name": "blue", "strength": 0.005}, "mode": "2d"},
    {"model_name": "gsdrunet",          "gmap_strategy": "gradient",   "unsharp_cfg": {"name": "unsharp", "amount": 3.0}, "dither_cfg": {"name": "blue",     "strength": 0.005}, "mode": "2d"},
    {"model_name": "restormer",         "gmap_strategy": "local_var_5","unsharp_cfg": {"name": "mlvum",   "scale": 3.0}, "dither_cfg": {"name": "gaussian", "strength": 0.005}, "mode": "2d"},
    {"model_name": "restormer",         "gmap_strategy": "local_var_9","unsharp_cfg": {"name": "mlvum",   "scale": 3.0}, "dither_cfg": {"name": "gaussian", "strength": 0.005}, "mode": "2d"},
]

SIGMAS = [0.1, 0.2]
OUT_PNG = _PROJ_ROOT / "benchmark_results" / "search" / "top10_results.png"


def short_label(cfg: dict) -> str:
    m = cfg["model_name"].replace("_pretrained", "").replace("drunet", "DRUNet").replace("gsdrunet", "GSDRUNet").replace("restormer", "Restormer")
    g = cfg["gmap_strategy"]
    u = cfg["unsharp_cfg"]["name"]
    d = cfg["dither_cfg"]["name"]
    return f"{m}\n{g} | {u} | {d}"


def run_and_capture(cfg: dict, sigma: float):
    """Run a trial and also return the actual images for visualization."""
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    rng = np.random.default_rng(42)
    clean = load_clean_2d()
    noisy = add_rician_noise(clean, sigma, rng)
    x = build_input_tensor(noisy, sigma, cfg["gmap_strategy"], "2d")

    model = _load_model(cfg["model_name"], cfg["gmap_strategy"])
    denoised = _run_model(model, x, sigma, cfg["model_name"])

    pp = apply_unsharp(denoised, cfg["unsharp_cfg"])
    final = apply_dither(pp, sigma, cfg["dither_cfg"])
    final = final.clip(0, 1)

    psnr = float(peak_signal_noise_ratio(clean, final, data_range=1.0))
    ssim = float(structural_similarity(clean, final, data_range=1.0))

    return psnr, ssim, clean, noisy, denoised, final


def main():
    print(f"Running {len(TOP10)} configs × {len(SIGMAS)} sigmas = {len(TOP10)*len(SIGMAS)} trials...\n")

    results = {}  # (cfg_idx, sigma) -> (psnr, ssim, clean, noisy, denoised, final)

    for i, cfg in enumerate(TOP10):
        for sigma in SIGMAS:
            label = f"[{i+1:2d}/10] s={sigma}  {cfg['model_name']:20s}"
            print(f"{label} ...", end=" ", flush=True)
            psnr, ssim, clean, noisy, denoised, final = run_and_capture(cfg, sigma)
            results[(i, sigma)] = (psnr, ssim, clean, noisy, denoised, final)
            print(f"PSNR={psnr:.2f} dB  SSIM={ssim:.4f}")

    # ── Build large plot ─────────────────────────────────────────────────────
    # Layout:
    #   Top section: bar chart (PSNR) for σ=0.1 and σ=0.2 side by side
    #   Middle section: bar chart (SSIM) for both sigmas
    #   Bottom section: image strip — for each sigma, show clean / noisy / best / worst

    labels = [short_label(c) for c in TOP10]
    x_pos  = np.arange(len(TOP10))

    psnr_01 = [results[(i, 0.1)][0] for i in range(10)]
    psnr_02 = [results[(i, 0.2)][0] for i in range(10)]
    ssim_01 = [results[(i, 0.1)][1] for i in range(10)]
    ssim_02 = [results[(i, 0.2)][1] for i in range(10)]

    fig = plt.figure(figsize=(28, 22), dpi=120)
    fig.patch.set_facecolor("#0f1117")
    cmap_bg = "#0f1117"
    cmap_ax = "#1a1d27"
    col_01  = "#4cc9f0"
    col_02  = "#f72585"
    col_txt = "#e0e0e0"
    col_grid= "#2a2d3a"

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1, 1, 1.1],
        hspace=0.5,
        wspace=0.25,
        left=0.06, right=0.97,
        top=0.93, bottom=0.04,
    )

    def style_ax(ax):
        ax.set_facecolor(cmap_ax)
        ax.tick_params(colors=col_txt, labelsize=7.5)
        ax.spines[:].set_color(col_grid)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(axis="y", color=col_grid, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    # ── PSNR bars ──
    for col, (sigma, vals, color) in enumerate([(0.1, psnr_01, col_01), (0.2, psnr_02, col_02)]):
        ax = fig.add_subplot(gs[0, col])
        style_ax(ax)
        bars = ax.bar(x_pos, vals, color=color, alpha=0.85, width=0.6, zorder=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5, color=col_txt)
        ax.set_ylabel("PSNR (dB)", color=col_txt, fontsize=9)
        ax.set_title(f"PSNR — σ = {sigma}", color=col_txt, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(min(vals) - 1, max(vals) + 2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=col_txt)
        # Highlight best
        best_i = int(np.argmax(vals))
        bars[best_i].set_edgecolor("gold")
        bars[best_i].set_linewidth(2)

    # ── SSIM bars ──
    for col, (sigma, vals, color) in enumerate([(0.1, ssim_01, col_01), (0.2, ssim_02, col_02)]):
        ax = fig.add_subplot(gs[1, col])
        style_ax(ax)
        bars = ax.bar(x_pos, vals, color=color, alpha=0.85, width=0.6, zorder=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5, color=col_txt)
        ax.set_ylabel("SSIM", color=col_txt, fontsize=9)
        ax.set_title(f"SSIM — σ = {sigma}", color=col_txt, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.05))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=col_txt)
        best_i = int(np.argmax(vals))
        bars[best_i].set_edgecolor("gold")
        bars[best_i].set_linewidth(2)

    # ── Image strips — best & worst pipeline for each sigma ──
    gs_img = gridspec.GridSpecFromSubplotSpec(
        2, 5, subplot_spec=gs[2, :],
        wspace=0.04, hspace=0.25,
    )

    def crop_center(img, size=160):
        h, w = img.shape
        r = (h - size) // 2
        c = (w - size) // 2
        return img[r:r+size, c:c+size]

    for row, sigma in enumerate(SIGMAS):
        psnr_row = [results[(i, sigma)][0] for i in range(10)]
        best_i  = int(np.argmax(psnr_row))
        worst_i = int(np.argmin(psnr_row))

        _, _, clean, noisy, _, final_best  = results[(best_i,  sigma)]
        _, _, _,     _,     _, final_worst = results[(worst_i, sigma)]

        imgs = [
            (crop_center(clean),       "Clean reference"),
            (crop_center(noisy),       f"Noisy  (σ={sigma})"),
            (crop_center(final_best),  f"Best #{best_i+1}\n{results[(best_i,sigma)][0]:.2f} dB"),
            (crop_center(final_worst), f"Worst #{worst_i+1}\n{results[(worst_i,sigma)][0]:.2f} dB"),
        ]

        # Difference map (best - clean, amplified)
        diff = np.abs(crop_center(final_best) - crop_center(clean)) * 5
        imgs.append((diff.clip(0, 1), f"Best residual ×5\nσ={sigma}"))

        for col, (img, title) in enumerate(imgs):
            ax = fig.add_subplot(gs_img[row, col])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, color=col_txt, fontsize=7.5, pad=3)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(col_grid)

    fig.suptitle(
        "Top-10 Denoising Pipeline Configurations — Benchmark Results",
        color=col_txt, fontsize=14, fontweight="bold", y=0.97,
    )

    fig.text(0.5, 0.006,
             "Gold outline = best per metric  |  Blue = σ=0.1  |  Pink = σ=0.2",
             ha="center", color="#888", fontsize=8)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PNG), dpi=120, bbox_inches="tight", facecolor=cmap_bg)
    plt.close(fig)
    print(f"\nSaved → {OUT_PNG}")


if __name__ == "__main__":
    main()
