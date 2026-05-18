"""
Classical noise-map benchmark for MRI denoising.

Loads 2-D DICOM slices, adds synthetic Rician noise at configurable sigma
levels, estimates noise sigma with multiple classical estimators, and
produces a comparison CSV and bar-chart figure.

Usage
-----
    python scripts/classical_noise_map.py \\
        --data C:/projetos/Datasets/dcm_BB_noisy \\
        --noise-levels 0.02 0.05 0.10 \\
        --output C:/projetos/MRI_Denoise/artifacts/noise_map

Outputs (inside --output)
--------------------------
    noise_map_results.csv  — columns: sigma_gt, filter, sigma_estimated, abs_error
    noise_map_bar.png      — grouped bar chart of abs_error per estimator & sigma level
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Make sure mr4raw filters are importable when the script is executed from the
# MRI_Denoise project root.
_MRI_DENOISE_ROOT = Path(__file__).resolve().parent.parent
_MR4RAW_SRC = _MRI_DENOISE_ROOT.parent / "mr4raw" / "src"
if str(_MR4RAW_SRC) not in sys.path:
    sys.path.insert(0, str(_MR4RAW_SRC))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise synthesis (identical formula to benchmark_pretrained.py)
# ---------------------------------------------------------------------------

def add_rician_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Rician noise to *img*.

    Rician model: ``sqrt((x + n1)^2 + n2^2)`` where n1, n2 ~ N(0, sigma).

    Parameters
    ----------
    img : np.ndarray
        Clean image, float32, values in [0, 1].
    sigma : float
        Noise standard deviation.

    Returns
    -------
    np.ndarray
        Noisy image, same shape and dtype as *img*, values >= 0.
    """
    n1 = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    n2 = np.random.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.sqrt((img + n1) ** 2 + n2 ** 2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dicom_slices(dicom_dir: Path, max_slices: int = 20) -> list[np.ndarray]:
    """Load DICOM slices from *dicom_dir*, normalised to [0, 1].

    Parameters
    ----------
    dicom_dir : Path
        Directory (searched recursively) containing ``*.dcm`` files.
    max_slices : int, optional
        Maximum number of slices to load (default 20).

    Returns
    -------
    list of np.ndarray
        Each element is a 2-D float32 array with values in [0, 1].
    """
    import pydicom  # lazy import — optional dependency

    files = sorted(dicom_dir.rglob("*.dcm"))[:max_slices]
    if not files:
        # Fall back to files without extension (some DICOMs lack one)
        files = [f for f in sorted(dicom_dir.rglob("*")) if f.is_file()][:max_slices]

    slices: list[np.ndarray] = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
            arr = ds.pixel_array.astype(np.float32)
            denom = arr.max() - arr.min()
            arr = (arr - arr.min()) / (denom + 1e-8)
            slices.append(arr)
        except Exception as exc:
            log.warning("Skipping %s: %s", f, exc)
    return slices


# ---------------------------------------------------------------------------
# Sigma estimators
# ---------------------------------------------------------------------------

def _estimate_mad(noisy: np.ndarray) -> float:
    """MAD-based sigma estimator (wraps ``filters.classical_filters.mad_sigma_estimate``)."""
    from filters.classical_filters import mad_sigma_estimate
    return mad_sigma_estimate(noisy)


def _estimate_wavelet(noisy: np.ndarray) -> float:
    """Wavelet-threshold sigma estimator.

    Uses ``skimage.restoration.estimate_sigma`` on the finest wavelet detail
    sub-band — a fast surrogate for the full wavelet decomposition.
    """
    from skimage.restoration import estimate_sigma
    return float(np.mean(estimate_sigma(noisy, channel_axis=None)))


def _estimate_nlm(noisy: np.ndarray) -> float:
    """NLM-based sigma estimator via ``skimage.restoration.estimate_sigma``."""
    from skimage.restoration import estimate_sigma
    return float(np.mean(estimate_sigma(noisy, channel_axis=None)))


def _estimate_bilateral(noisy: np.ndarray) -> float:
    """Bilateral-residual sigma estimator.

    Denoises with a bilateral filter and treats the std of the residual as
    the estimated sigma.
    """
    import cv2
    denoised = cv2.bilateralFilter(noisy.astype(np.float32), d=5,
                                   sigmaColor=0.1, sigmaSpace=5)
    residual = noisy.astype(np.float64) - denoised.astype(np.float64)
    return float(np.std(residual))


def _estimate_tv(noisy: np.ndarray) -> float:
    """TV-residual sigma estimator.

    Denoises with TV Chambolle and treats the std of the residual as the
    estimated sigma.
    """
    from skimage.restoration import denoise_tv_chambolle
    denoised = denoise_tv_chambolle(noisy, weight=0.1)
    residual = noisy.astype(np.float64) - denoised.astype(np.float64)
    return float(np.std(residual))


# Registry: name -> callable(noisy_image) -> float
ESTIMATORS: dict[str, object] = {
    "mad": _estimate_mad,
    "wavelet": _estimate_wavelet,
    "nlm": _estimate_nlm,
    "bilateral": _estimate_bilateral,
    "tv": _estimate_tv,
}


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_noise_map(
    slices: list[np.ndarray],
    noise_levels: Sequence[float],
) -> list[dict]:
    """Estimate noise sigma for each (slice, noise_level, estimator) combination.

    Parameters
    ----------
    slices : list of np.ndarray
        Clean 2-D float32 images in [0, 1].
    noise_levels : sequence of float
        Ground-truth sigma values to test.

    Returns
    -------
    list of dict
        Each dict has keys: sigma_gt, filter, sigma_estimated, abs_error.
    """
    records: list[dict] = []

    for sigma_gt in noise_levels:
        log.info("Noise level sigma=%.3f", sigma_gt)
        for slice_idx, clean in enumerate(slices):
            noisy = np.clip(add_rician_noise(clean, sigma_gt), 0.0, 1.0)

            for est_name, est_fn in ESTIMATORS.items():
                try:
                    sigma_est = float(est_fn(noisy))  # type: ignore[operator]
                except Exception as exc:
                    log.warning("  Estimator %s failed on slice %d: %s",
                                est_name, slice_idx, exc)
                    sigma_est = float("nan")

                records.append({
                    "sigma_gt": sigma_gt,
                    "filter": est_name,
                    "sigma_estimated": sigma_est,
                    "abs_error": abs(sigma_est - sigma_gt),
                })

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_results(records: list[dict], output_dir: Path) -> None:
    """Save CSV and bar-chart figure to *output_dir*.

    Parameters
    ----------
    records : list of dict
        Output of :func:`run_noise_map`.
    output_dir : Path
        Directory where outputs are written (created if absent).
    """
    import pandas as pd  # lazy import
    import matplotlib  # lazy import
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    # --- CSV -----------------------------------------------------------------
    csv_path = output_dir / "noise_map_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved CSV: %s", csv_path)

    # --- Bar chart -----------------------------------------------------------
    # Mean abs_error per (filter, sigma_gt)
    summary = (
        df.groupby(["sigma_gt", "filter"])["abs_error"]
        .mean()
        .reset_index()
    )

    sigma_values = sorted(summary["sigma_gt"].unique())
    estimator_names = sorted(summary["filter"].unique())
    n_sigmas = len(sigma_values)
    n_estimators = len(estimator_names)

    x = np.arange(n_sigmas)
    bar_width = 0.8 / max(n_estimators, 1)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_sigmas * n_estimators), 5))

    for idx, est in enumerate(estimator_names):
        subset = summary[summary["filter"] == est]
        # Align values with sigma_values order
        heights = [
            float(subset.loc[subset["sigma_gt"] == sv, "abs_error"].iloc[0])
            if sv in subset["sigma_gt"].values else float("nan")
            for sv in sigma_values
        ]
        offset = (idx - n_estimators / 2.0 + 0.5) * bar_width
        ax.bar(x + offset, heights, width=bar_width * 0.9, label=est)

    ax.set_xlabel("Ground-truth sigma")
    ax.set_ylabel("Mean absolute error of sigma estimate")
    ax.set_title("Noise sigma estimation: |sigma_estimated - sigma_gt|")
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ={sv:.3f}" for sv in sigma_values])
    ax.legend(title="Estimator")
    plt.tight_layout()

    chart_path = output_dir / "noise_map_bar.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved chart: %s", chart_path)


# ---------------------------------------------------------------------------
# Public entry point (importable by tests)
# ---------------------------------------------------------------------------

def generate_noise_map(
    data_dir: Path,
    noise_levels: Sequence[float],
    output_dir: Path,
    max_slices: int = 20,
) -> list[dict]:
    """End-to-end noise-map pipeline.

    Parameters
    ----------
    data_dir : Path
        Directory containing DICOM files.
    noise_levels : sequence of float
        Ground-truth sigma values.
    output_dir : Path
        Where CSV and PNG are written.
    max_slices : int, optional
        Maximum number of DICOM slices to load (default 20).

    Returns
    -------
    list of dict
        Raw records (also persisted to CSV).
    """
    log.info("Loading DICOM slices from %s", data_dir)
    slices = load_dicom_slices(data_dir, max_slices=max_slices)
    if not slices:
        raise RuntimeError(f"No DICOM slices found in {data_dir}")
    log.info("Loaded %d slices", len(slices))

    records = run_noise_map(slices, noise_levels)
    save_results(records, output_dir)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical noise-sigma estimation benchmark for MRI DICOMs."
    )
    p.add_argument(
        "--data", type=Path, required=True,
        help="Directory containing DICOM files (*.dcm, searched recursively).",
    )
    p.add_argument(
        "--noise-levels", nargs="+", type=float, default=[0.02, 0.05, 0.10],
        metavar="SIGMA",
        help="Ground-truth Rician noise sigma levels (default: 0.02 0.05 0.10).",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path("C:/projetos/MRI_Denoise/artifacts/noise_map"),
        help="Output directory for CSV and PNG.",
    )
    p.add_argument(
        "--max-slices", type=int, default=20,
        help="Maximum number of DICOM slices to load (default 20).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    generate_noise_map(
        data_dir=args.data,
        noise_levels=args.noise_levels,
        output_dir=args.output,
        max_slices=args.max_slices,
    )


if __name__ == "__main__":
    main()
