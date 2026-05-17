"""
Comprehensive MRI Denoising Benchmark — Track A (2D) + Track B (3D).

Runs all pretrained denoisers on IXI, Lumbar, and MR4Raw datasets across
four Rician noise levels. Two parallel tracks:
  - Track A: 2D slice-by-slice for all pretrained models
  - Track B: 3D full-volume for native 3D models (RicianNet3D)

Usage
-----
    python scripts/comprehensive_benchmark.py --output C:/projetos/artifacts/benchmark/2026-05-17
    python scripts/comprehensive_benchmark.py --skip-3d --models drunet_pretrained,gsdrunet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import from benchmark_pretrained (avoids circular at module load time)
# ---------------------------------------------------------------------------

def _bp():
    import benchmark_pretrained as _m
    # benchmark_pretrained inserts mr4raw/src before MRI_Denoise/src, which
    # causes `from models import get_model` to find the wrong package.
    # Re-insert MRI_Denoise/src at front so the right models package wins.
    _src = str(_ROOT / "src")
    if sys.path[0] != _src:
        try:
            sys.path.remove(_src)
        except ValueError:
            pass
        sys.path.insert(0, _src)
    # Drop cached models entry if it's from the wrong path
    import importlib
    if "models" in sys.modules:
        cached_path = getattr(sys.modules["models"], "__file__", "") or ""
        if _src not in cached_path:
            del sys.modules["models"]
    return _m


TRACK_A_MODELS = [
    "drunet_pretrained", "gsdrunet", "restormer", "swinir_pretrained",
    "dncnn_pretrained", "scunet_pretrained", "nafnet_xs", "unet",
    "ffdnet", "snraware", "bm3d",
]
TRACK_B_MODELS = ["3d-parallel-ricianet"]

DEFAULT_SIGMAS = [0.02, 0.05, 0.10, 0.20]
DEFAULT_OUTPUT = Path("artifacts") / "benchmark" / str(date.today())


# ---------------------------------------------------------------------------
# Progress checkpoint helpers
# ---------------------------------------------------------------------------

def _progress_key(track: str, model: str, dataset: str, modality: str,
                  sigma: float) -> str:
    return f"{track}|{model}|{dataset}|{modality}|{sigma}"


def load_progress(progress_file: Path) -> set:
    if progress_file.exists():
        return set(json.loads(progress_file.read_text(encoding="utf-8")))
    return set()


def save_progress(progress_file: Path, done: set) -> None:
    progress_file.write_text(json.dumps(sorted(done)), encoding="utf-8")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics_2d(denoised_slices: list[np.ndarray],
                         clean_slices: list[np.ndarray],
                         device: torch.device) -> dict:
    bp = _bp()
    psnr_v, ssim_v, haar_v, vgg_v, sharp_v = [], [], [], [], []
    for d, c in zip(denoised_slices, clean_slices):
        d_t = torch.tensor(d).unsqueeze(0).unsqueeze(0)
        c_t = torch.tensor(c).unsqueeze(0).unsqueeze(0)
        psnr_v.append(bp.psnr(d_t, c_t))
        ssim_v.append(bp.ssim_2d(d_t.squeeze(), c_t.squeeze()))
        haar_v.append(bp.haarpsi(d_t.squeeze(0), c_t.squeeze(0)))
        vgg_v.append(bp.vgg_sim(d_t, c_t, device))
        sharp_v.append(bp.sharpness(d_t))
    with np.errstate(all="ignore"):
        return {
            "psnr": float(np.nanmean(psnr_v)),
            "ssim": float(np.nanmean(ssim_v)),
            "haarpsi": float(np.nanmean(haar_v)),
            "vgg_loss": float(np.nanmean(vgg_v)),
            "sharpness": float(np.nanmean(sharp_v)),
        }


# ---------------------------------------------------------------------------
# Track A — 2D
# ---------------------------------------------------------------------------

def run_benchmark_2d(
    loaders: list,
    models: dict,
    sigma_levels: list[float],
    device: torch.device,
    progress: set,
) -> list[dict]:
    """Run 2D slice-by-slice benchmark. Returns list of record dicts."""
    bp = _bp()
    records: list[dict] = []

    for loader in loaders:
        for item in loader.volumes():
            vol = item["volume"]  # (H, W, D)
            D = vol.shape[2]
            clean_slices = [vol[:, :, k] for k in range(D)]
            meta = {
                "dataset": item["dataset"],
                "modality": item["modality"],
                "subject_id": item["subject_id"],
            }

            for sigma in sigma_levels:
                noisy_vol = bp.add_rician_noise(vol, sigma)
                noisy_slices = [np.clip(noisy_vol[:, :, k], 0.0, 1.0)
                                for k in range(D)]

                # Noisy baseline
                key = _progress_key("noisy", "noisy_input",
                                    item["dataset"], item["modality"], sigma)
                if key not in progress:
                    m = _compute_metrics_2d(noisy_slices, clean_slices, device)
                    records.append({"track": "2d", "model": "noisy_input",
                                    **meta, "sigma": sigma, **m})
                    progress.add(key)

                for model_name, model in models.items():
                    key = _progress_key("2d", model_name,
                                        item["dataset"], item["modality"], sigma)
                    if key in progress:
                        continue
                    try:
                        denoised = [
                            np.clip(bp.run_model(model, s, sigma, device), 0.0, 1.0)
                            for s in noisy_slices
                        ]
                        m = _compute_metrics_2d(denoised, clean_slices, device)
                        records.append({"track": "2d", "model": model_name,
                                        **meta, "sigma": sigma, **m})
                        progress.add(key)
                        log.info(
                            f"  [2D] {model_name} | {item['dataset']} "
                            f"{item['modality']} σ={sigma:.2f} "
                            f"PSNR={m['psnr']:.2f} dB"
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            log.warning(f"  [2D] {model_name} OOM — retrying on CPU")
                            try:
                                cpu_dev = torch.device("cpu")
                                model_cpu = model.cpu()
                                denoised = [
                                    np.clip(bp.run_model(model_cpu, s, sigma, cpu_dev),
                                            0.0, 1.0)
                                    for s in noisy_slices
                                ]
                                m = _compute_metrics_2d(denoised, clean_slices, cpu_dev)
                                records.append({"track": "2d", "model": model_name,
                                                **meta, "sigma": sigma, **m})
                                progress.add(key)
                                model.to(device)
                            except Exception as e2:
                                log.warning(
                                    f"  [2D] {model_name} CPU fallback failed: {e2}"
                                )
                        else:
                            log.warning(f"  [2D] {model_name} failed: {e}")
                    except Exception as e:
                        log.warning(f"  [2D] {model_name} failed: {e}")

    return records


# ---------------------------------------------------------------------------
# Track B — 3D
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_model_3d(model: torch.nn.Module, noisy_vol: np.ndarray,
                 sigma: float, device: torch.device) -> np.ndarray:
    """Run a 3D model on a full volume.

    Args:
        noisy_vol: (H, W, D) float32 in [0, 1]
    Returns:
        denoised: (D, H, W) float32
    """
    vol_dhw = np.transpose(noisy_vol, (2, 0, 1))  # (D, H, W)
    t = torch.from_numpy(vol_dhw).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    s = torch.full_like(t, sigma)
    inp = torch.cat([t, s], dim=1)  # (1,2,D,H,W)
    out = model(inp)
    return out.squeeze().cpu().numpy()  # (D, H, W)


def _compute_metrics_3d(denoised_dhw: np.ndarray, clean_hwD: np.ndarray,
                         device: torch.device) -> dict:
    D = clean_hwD.shape[2]
    denoised_slices = [denoised_dhw[k, :, :] for k in range(D)]
    clean_slices = [clean_hwD[:, :, k] for k in range(D)]
    return _compute_metrics_2d(denoised_slices, clean_slices, device)


def run_benchmark_3d(
    loaders: list,
    models: dict,
    sigma_levels: list[float],
    device: torch.device,
    progress: set,
) -> list[dict]:
    """Run 3D full-volume benchmark. Returns list of record dicts."""
    bp = _bp()
    records: list[dict] = []

    for loader in loaders:
        for item in loader.volumes():
            vol = item["volume"]  # (H, W, D)
            meta = {
                "dataset": item["dataset"],
                "modality": item["modality"],
                "subject_id": item["subject_id"],
            }
            for sigma in sigma_levels:
                noisy_vol = np.clip(bp.add_rician_noise(vol, sigma), 0.0, 1.0)
                for model_name, model in models.items():
                    key = _progress_key("3d", model_name,
                                        item["dataset"], item["modality"], sigma)
                    if key in progress:
                        continue
                    try:
                        denoised = np.clip(
                            run_model_3d(model, noisy_vol, sigma, device), 0.0, 1.0
                        )
                        m = _compute_metrics_3d(denoised, vol, device)
                        records.append({"track": "3d", "model": model_name,
                                        **meta, "sigma": sigma, **m})
                        progress.add(key)
                        log.info(
                            f"  [3D] {model_name} | {item['dataset']} "
                            f"{item['modality']} σ={sigma:.2f} "
                            f"PSNR={m['psnr']:.2f} dB"
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            log.warning(f"  [3D] {model_name} OOM — skipping")
                        else:
                            log.warning(f"  [3D] {model_name} failed: {e}")
                    except Exception as e:
                        log.warning(f"  [3D] {model_name} failed: {e}")
    return records


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models_for_track(model_names: list[str], device: torch.device,
                           skipped_file: Path) -> dict:
    """Load all models; skip failures. Returns {name: model}."""
    bp = _bp()
    models = {}
    skipped = []
    for name in model_names:
        m = bp.load_model(name, device)
        if m is not None:
            models[name] = m
        else:
            skipped.append({"model": name, "status": "skipped",
                            "reason": "load_model returned None"})
    if skipped:
        skipped_file.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if skipped_file.exists():
            existing = json.loads(skipped_file.read_text(encoding="utf-8"))
        skipped_file.write_text(
            json.dumps(existing + skipped, indent=2), encoding="utf-8"
        )
    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive MRI denoising benchmark"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--sigma-levels", nargs="+", type=float, default=DEFAULT_SIGMAS
    )
    parser.add_argument(
        "--max-volumes", type=int, default=1,
        help="Volumes per modality per dataset (default 1)"
    )
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated model keys or 'all'"
    )
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing .progress.json"
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Checkpoint
    progress_file = output_dir / ".progress.json"
    progress = set() if args.no_resume else load_progress(progress_file)
    if progress:
        log.info(f"Resuming: {len(progress)} entries already done")

    # Model selection
    track_a_names = TRACK_A_MODELS
    if args.models != "all":
        track_a_names = [m.strip() for m in args.models.split(",")]

    # Dataset loaders
    from data.benchmark_loaders import IXILoader, LumbarLoader, MR4RawLoader
    loaders = [IXILoader(), LumbarLoader(), MR4RawLoader()]

    # Track A — 2D
    log.info("=== Track A — 2D ===")
    models_2d = load_models_for_track(
        track_a_names, device, output_dir / "skipped_models.json"
    )
    records_2d = run_benchmark_2d(loaders, models_2d, args.sigma_levels,
                                   device, progress)
    save_progress(progress_file, progress)

    # Track B — 3D
    records_3d = []
    if not args.skip_3d:
        log.info("=== Track B — 3D ===")
        models_3d = load_models_for_track(
            TRACK_B_MODELS, device, output_dir / "skipped_models.json"
        )
        if models_3d:
            records_3d = run_benchmark_3d(loaders, models_3d,
                                           args.sigma_levels, device, progress)
            save_progress(progress_file, progress)
        else:
            log.info("  No 3D models loaded — skipping Track B")

    # Combine and save
    all_records = records_2d + records_3d
    if not all_records:
        log.warning("No records produced — check model loading and datasets")
        return

    df = pd.DataFrame(all_records)
    csv_path = output_dir / "benchmark_results.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(
            subset=["track", "model", "dataset", "modality", "subject_id", "sigma"]
        )
    df.to_csv(csv_path, index=False)
    log.info(f"Saved: {csv_path}  ({len(df)} rows)")

    # Leaderboards
    for track in ("2d", "3d"):
        sub = df[df["track"] == track]
        if sub.empty:
            continue
        noisy_mask = sub["model"] == "noisy_input"
        lb = (
            sub[~noisy_mask]
            .groupby(["model", "dataset", "modality", "sigma"])[
                ["psnr", "ssim", "haarpsi", "vgg_loss", "sharpness"]
            ]
            .mean()
            .reset_index()
            .sort_values("psnr", ascending=False)
        )
        lb_path = output_dir / f"leaderboard_{track}.csv"
        lb.to_csv(lb_path, index=False)
        log.info(f"Saved: {lb_path}")
        log.info(
            f"\n=== Leaderboard {track.upper()} (top 5 by PSNR) ===\n"
            + lb.head(5).to_string(index=False)
        )

    # Report
    try:
        import importlib.util as _ilu
        _cr_spec = _ilu.spec_from_file_location(
            "comprehensive_report",
            Path(__file__).parent / "comprehensive_report.py",
        )
        _cr = _ilu.module_from_spec(_cr_spec)
        _cr_spec.loader.exec_module(_cr)
        _cr.generate_comprehensive_report(csv_path, output_dir)
    except Exception as e:
        log.warning(f"Report generation failed: {e}")

    # Gemini summary
    try:
        sys.path.insert(0, str(_ROOT.parent))
        from gemini_agent import spawn
        lb_2d_path = output_dir / "leaderboard_2d.csv"
        context_files = [lb_2d_path] if lb_2d_path.exists() else []
        summary = spawn(
            "Rank the MRI denoisers by PSNR/SSIM/HaarPSI trade-off and recommend "
            "the best model for clinical magnitude MRI denoising. Consider performance "
            "across datasets and noise levels.",
            context_files=context_files,
            complexity=4,
        )
        rec_path = output_dir / "recommendation.md"
        rec_path.write_text(f"# Denoiser Recommendation\n\n{summary}\n",
                            encoding="utf-8")
        log.info(f"Saved: {rec_path}")
    except Exception as e:
        log.warning(f"Gemini summary failed: {e}")


if __name__ == "__main__":
    main()
