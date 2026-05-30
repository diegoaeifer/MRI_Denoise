"""Top-10 denoising pipelines by SSIM from results.jsonl.

For each sigma (0.1, 0.2), produces a separate plot with 10 rows (one per config)
and 4 columns: Original | Noisy | Denoised | Residual (×5).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

RESULTS_JSONL = _PROJ_ROOT / "benchmark_results" / "search" / "results.jsonl"
OUT_DIR       = _PROJ_ROOT / "benchmark_results" / "search"
SIGMAS        = [0.1, 0.2]


def load_top10_by_ssim(sigma: float) -> list[dict]:
    entries = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("error") or obj.get("ssim") is None:
            continue
        if abs(obj["sigma"] - sigma) < 1e-6:
            entries.append(obj)
    entries.sort(key=lambda x: x["ssim"], reverse=True)
    # deduplicate by (model, gmap, unsharp.name, dither.name, mode)
    seen, top10 = set(), []
    for e in entries:
        key = (e["model_name"], e["gmap_strategy"],
               e["unsharp_cfg"]["name"], e["dither_cfg"]["name"], e["mode"])
        if key not in seen:
            seen.add(key)
            top10.append(e)
        if len(top10) == 10:
            break
    return top10


def run_and_capture(cfg: dict, sigma: float):
    rng = np.random.default_rng(42)
    clean = load_clean_2d()
    noisy = add_rician_noise(clean, sigma, rng)
    x = build_input_tensor(noisy, sigma, cfg["gmap_strategy"], cfg["mode"])

    model = _load_model(cfg["model_name"], cfg["gmap_strategy"])
    denoised_raw = _run_model(model, x, sigma, cfg["model_name"])

    pp    = apply_unsharp(denoised_raw, cfg["unsharp_cfg"])
    final = apply_dither(pp, sigma, cfg["dither_cfg"]).clip(0, 1)

    psnr = float(peak_signal_noise_ratio(clean, final, data_range=1.0))
    ssim = float(structural_similarity(clean, final, data_range=1.0))
    residual = np.abs(final - clean)

    return clean, noisy, final, residual, psnr, ssim


def short_label(cfg: dict) -> str:
    m = (cfg["model_name"]
         .replace("_pretrained", "")
         .replace("drunet", "DRUNet")
         .replace("dncnn", "DnCNN")
         .replace("gsdrunet", "GSDRUNet")
         .replace("restormer", "Restormer")
         .replace("swinir", "SwinIR"))
    return f"{m}  |  gmap={cfg['gmap_strategy']}  unsharp={cfg['unsharp_cfg']['name']}  dither={cfg['dither_cfg']['name']}"


def make_plot(sigma: float):
    print(f"\n=== sigma = {sigma} ===")
    top10 = load_top10_by_ssim(sigma)

    print(f"Running {len(top10)} configs...")
    rows_data = []
    for i, cfg in enumerate(top10):
        print(f"  [{i+1}/10] {cfg['model_name']:25s} SSIM_stored={cfg['ssim']:.4f} ...", end=" ", flush=True)
        clean, noisy, final, residual, psnr, ssim = run_and_capture(cfg, sigma)
        rows_data.append((cfg, clean, noisy, final, residual, psnr, ssim))
        print(f"PSNR={psnr:.2f} dB  SSIM={ssim:.4f}")

    # ── Build plot ─────────────────────────────────────────────────────────
    BG  = "#0f1117"
    AX  = "#1a1d27"
    GRD = "#2a2d3a"
    TXT = "#e0e0e0"
    COL_HEADERS = ["Original", "Noisy", "Denoised", "Residual ×5"]
    COL_CMAPS   = ["gray", "gray", "gray", "inferno"]

    n_rows = len(rows_data)
    fig = plt.figure(figsize=(18, n_rows * 2.6 + 1.5), dpi=110)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        n_rows, 4,
        figure=fig,
        wspace=0.04, hspace=0.35,
        left=0.22, right=0.98,
        top=0.95, bottom=0.02,
    )

    for row, (cfg, clean, noisy, final, residual, psnr, ssim) in enumerate(rows_data):
        imgs = [clean, noisy, final, (residual * 5).clip(0, 1)]
        for col, (img, cmap) in enumerate(zip(imgs, COL_CMAPS)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(COL_HEADERS[col], color=TXT, fontsize=10,
                             fontweight="bold", pad=6)
            if col == 3:
                ax.text(1.02, 0.5,
                        f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}",
                        transform=ax.transAxes,
                        color=TXT, fontsize=8, va="center",
                        bbox=dict(facecolor=AX, edgecolor=GRD, boxstyle="round,pad=0.3"))

        # Row label on the left
        fig.text(
            0.01, 1 - (row + 0.5) / n_rows * (1 - 0.02 - 0.05) - 0.05,
            f"#{row+1}\n{short_label(cfg)}",
            color=TXT, fontsize=7, va="center", ha="left",
            transform=fig.transFigure,
        )

    fig.suptitle(
        f"Top-10 Pipelines by SSIM — sigma = {sigma}",
        color=TXT, fontsize=13, fontweight="bold", y=0.98,
    )

    out_path = OUT_DIR / f"top10_ssim_visual_sigma{int(sigma*10):02d}.png"
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [make_plot(s) for s in SIGMAS]
    return paths


if __name__ == "__main__":
    main()
