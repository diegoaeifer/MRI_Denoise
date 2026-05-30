"""Benchmark SNRAware and ImT-MRD across 4 scaling factors at sigma=0.1 and sigma=0.2.

SNRAware scaling_factor: 0.25, 0.5, 0.75, 1.0  (smaller = stronger denoising)
ImT-MRD input_scale:    0.25, 0.5, 1.0, 2.0    (pre/post-scale, same principle)

Produces a large plot: image grid per model/sigma + PSNR/SSIM line charts.
"""
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
    load_clean_2d, add_rician_noise,
    build_input_tensor, _load_model, _run_model,
)
from search.postprocess import apply_unsharp
from search.dither import apply_dither
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

OUT_PNG = _PROJ_ROOT / "benchmark_results" / "search" / "scaling_factor_results.png"

# ── Model/scaling configurations ────────────────────────────────────────────

SNRAWARE_MODELS = ["snraware_small", "snraware_medium", "snraware_large"]
SNRAWARE_SCALES = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0]  # scaling_factor

IMTMRD_MODELS  = ["imt-mrd_residual", "imt-mrd_complex"]
IMTMRD_SCALES  = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 8.0]  # input_scale

SIGMAS = [0.1, 0.2]

# Best gmap/unsharp/dither from top-10 search
BEST_CFG = {
    "gmap_strategy": "wavelet",
    "unsharp_cfg": {"name": "mlvum", "scale": 3.0},
    "dither_cfg": {"name": "blue", "strength": 0.005},
    "mode": "2d",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_model_name(base: str, scale: float) -> str:
    """Convert base model name + scale to eval_pipeline key."""
    tag = int(round(scale * 100))
    if base.startswith("snraware_"):
        parts = base.split("_")
        return f"snraware_{parts[1]}_sf_{tag}"
    else:
        parts = base.split("-mrd_")
        return f"imt-mrd_{parts[1]}_is_{tag}"


def run_one(base_model: str, scale: float, sigma: float):
    rng = np.random.default_rng(42)
    clean = load_clean_2d()
    noisy = add_rician_noise(clean, sigma, rng)
    x = build_input_tensor(noisy, sigma, BEST_CFG["gmap_strategy"], "2d")

    model_key = make_model_name(base_model, scale)
    model = _load_model(model_key, BEST_CFG["gmap_strategy"])
    denoised = _run_model(model, x, sigma, model_key)

    pp    = apply_unsharp(denoised, BEST_CFG["unsharp_cfg"])
    final = apply_dither(pp, sigma, BEST_CFG["dither_cfg"]).clip(0, 1)

    psnr = float(peak_signal_noise_ratio(clean, final, data_range=1.0))
    ssim = float(structural_similarity(clean, final, data_range=1.0))
    return psnr, ssim, clean, noisy, final


def crop(img, size=128):
    h, w = img.shape
    r = (h - size) // 2
    c = (w - size) // 2
    return img[r:r+size, c:c+size]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_models  = SNRAWARE_MODELS + IMTMRD_MODELS
    scale_map   = {m: SNRAWARE_SCALES for m in SNRAWARE_MODELS}
    scale_map.update({m: IMTMRD_SCALES for m in IMTMRD_MODELS})

    # results[model][sigma][scale] = (psnr, ssim, clean, noisy, final)
    results = {m: {s: {} for s in SIGMAS} for m in all_models}

    total = sum(len(scale_map[m]) * len(SIGMAS) for m in all_models)
    done  = 0
    for model in all_models:
        for sigma in SIGMAS:
            for scale in scale_map[model]:
                done += 1
                tag = f"sf={scale}" if model.startswith("snraware") else f"is={scale}"
                print(f"[{done:2d}/{total}] {model:25s} {tag}  sigma={sigma} ...", end=" ", flush=True)
                psnr, ssim, clean, noisy, final = run_one(model, scale, sigma)
                results[model][sigma][scale] = (psnr, ssim, clean, noisy, final)
                print(f"PSNR={psnr:.2f} dB  SSIM={ssim:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    # Layout:
    #   Row 0-2  : SNRAware models — line charts (PSNR + SSIM) × 2 sigmas
    #   Row 3-4  : ImT-MRD models  — line charts
    #   Row 5-6  : Image strips    — one row per model group, columns = scales
    #              (sigma=0.1 top, sigma=0.2 bottom)

    BG  = "#0f1117"
    AX  = "#1a1d27"
    GRD = "#2a2d3a"
    TXT = "#e0e0e0"
    C01 = "#4cc9f0"   # sigma 0.1
    C02 = "#f72585"   # sigma 0.2
    MARKERS = "oDsP"

    MODEL_COLORS = {
        "snraware_small":   "#a8dadc",
        "snraware_medium":  "#457b9d",
        "snraware_large":   "#1d3557",
        "imt-mrd_residual": "#f4a261",
        "imt-mrd_complex":  "#e76f51",
    }
    MODEL_SHORT = {
        "snraware_small":   "SNRAware-S",
        "snraware_medium":  "SNRAware-M",
        "snraware_large":   "SNRAware-L",
        "imt-mrd_residual": "ImT-MRD Res",
        "imt-mrd_complex":  "ImT-MRD Cmplx",
    }

    fig = plt.figure(figsize=(30, 26), dpi=110)
    fig.patch.set_facecolor(BG)

    # GridSpec: top = 2 groups (SNRAware / ImT-MRD) of line charts
    #           bottom = image strips
    outer = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[1.4, 1.0, 1.4],
        hspace=0.45,
        left=0.05, right=0.97,
        top=0.94, bottom=0.03,
    )

    def style_ax(ax, ylabel="", title=""):
        ax.set_facecolor(AX)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.spines[:].set_color(GRD)
        ax.grid(color=GRD, linewidth=0.5, alpha=0.8)
        ax.set_axisbelow(True)
        if ylabel:
            ax.set_ylabel(ylabel, color=TXT, fontsize=9)
        if title:
            ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=6)
        ax.yaxis.label.set_color(TXT)
        ax.xaxis.label.set_color(TXT)

    # ── Line charts row 0: SNRAware PSNR/SSIM ──────────────────────────────
    gs_snr = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[0], wspace=0.35
    )

    def plot_lines(ax_psnr, ax_ssim, models, scale_key="sf"):
        for j, model in enumerate(models):
            scales = scale_map[model]
            for sigma, ls, color_s in [(0.1, "-", C01), (0.2, "--", C02)]:
                psnr_vals = [results[model][sigma][s][0] for s in scales]
                ssim_vals = [results[model][sigma][s][1] for s in scales]
                lbl = f"{MODEL_SHORT[model]} σ={sigma}"
                ax_psnr.plot(scales, psnr_vals, marker=MARKERS[j], linestyle=ls,
                             color=MODEL_COLORS[model], linewidth=1.8,
                             markersize=7, label=lbl,
                             markerfacecolor=color_s, markeredgecolor=MODEL_COLORS[model])
                ax_ssim.plot(scales, ssim_vals, marker=MARKERS[j], linestyle=ls,
                             color=MODEL_COLORS[model], linewidth=1.8,
                             markersize=7,
                             markerfacecolor=color_s, markeredgecolor=MODEL_COLORS[model])

    # SNRAware: 2 axes for PSNR, 2 for SSIM
    ax_snr_psnr = fig.add_subplot(gs_snr[0, :2])
    ax_snr_ssim = fig.add_subplot(gs_snr[0, 2:])
    style_ax(ax_snr_psnr, "PSNR (dB)", "SNRAware — PSNR vs scaling_factor")
    style_ax(ax_snr_ssim, "SSIM",      "SNRAware — SSIM vs scaling_factor")
    plot_lines(ax_snr_psnr, ax_snr_ssim, SNRAWARE_MODELS)
    ax_snr_psnr.set_xlabel("scaling_factor  (smaller = stronger)", color=TXT, fontsize=8)
    ax_snr_ssim.set_xlabel("scaling_factor  (smaller = stronger)", color=TXT, fontsize=8)
    ax_snr_psnr.legend(fontsize=7, facecolor=AX, labelcolor=TXT, framealpha=0.8)

    # ── Line charts row 1: ImT-MRD ──────────────────────────────────────────
    gs_imt = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1], wspace=0.35
    )
    ax_imt_psnr = fig.add_subplot(gs_imt[0, 0])
    ax_imt_ssim = fig.add_subplot(gs_imt[0, 1])
    style_ax(ax_imt_psnr, "PSNR (dB)", "ImT-MRD — PSNR vs input_scale")
    style_ax(ax_imt_ssim, "SSIM",      "ImT-MRD — SSIM vs input_scale")
    plot_lines(ax_imt_psnr, ax_imt_ssim, IMTMRD_MODELS, scale_key="is")
    ax_imt_psnr.set_xlabel("input_scale  (smaller = stronger)", color=TXT, fontsize=8)
    ax_imt_ssim.set_xlabel("input_scale  (smaller = stronger)", color=TXT, fontsize=8)
    ax_imt_psnr.legend(fontsize=7, facecolor=AX, labelcolor=TXT, framealpha=0.8)

    # ── Image strip row 2 ────────────────────────────────────────────────────
    # Columns: clean | noisy | snraware_small×4scales | imt-mrd_residual×4scales
    # But that's too many; show best model per group × 4 scales × 2 sigmas

    # Pick snraware_small and imt-mrd_residual as representatives
    # Rows: sigma=0.1 / sigma=0.2   Cols: clean, noisy, sf0.25, sf0.5, sf0.75, sf1.0
    REP_SNR = "snraware_small"
    REP_IMT = "imt-mrd_residual"

    # 10 columns: clean + noisy + 8 scales
    gs_img = gridspec.GridSpecFromSubplotSpec(
        4, 10, subplot_spec=outer[2],
        wspace=0.04, hspace=0.3,
    )

    for row_i, (model, scales, label) in enumerate([
        (REP_SNR, SNRAWARE_SCALES, "SNRAware-S"),
        (REP_IMT, IMTMRD_SCALES,   "ImT-MRD Residual"),
    ]):
        for row_s, sigma in enumerate(SIGMAS):
            row = row_i * 2 + row_s
            _, _, clean, noisy, _ = results[model][sigma][scales[0]]

            imgs = [crop(clean), crop(noisy)] + [
                crop(results[model][sigma][s][4]) for s in scales
            ]
            subtitles = ["Clean ref", f"Noisy σ={sigma}"] + [
                f"s={s}\n{results[model][sigma][s][0]:.2f}dB\n{results[model][sigma][s][1]:.3f}"
                for s in scales
            ]

            for col, (img, sub) in enumerate(zip(imgs, subtitles)):
                ax = fig.add_subplot(gs_img[row, col])
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(sub, color=TXT, fontsize=6, pad=2)
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(f"{label}\nσ={sigma}", color=TXT, fontsize=7,
                                  rotation=90, labelpad=4)
                    ax.axis("on")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines[:].set_color(GRD)

    fig.suptitle(
        "SNRAware & ImT-MRD — Scaling Factor Sweep  |  Best pipeline (wavelet gmap + mlvum + blue dither)",
        color=TXT, fontsize=13, fontweight="bold", y=0.975,
    )
    fig.text(0.5, 0.005,
             "Solid line = sigma 0.1  |  Dashed = sigma 0.2  |  Filled marker color: blue=0.1, pink=0.2",
             ha="center", color="#888", fontsize=8)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PNG), dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
