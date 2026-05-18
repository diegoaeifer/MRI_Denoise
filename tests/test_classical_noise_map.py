"""Tests for classical_noise_map.py and the new classical_filters extensions.

TDD: these tests are written to fail until the implementations are in place,
then to pass once they are.

Run with:
    pytest MRI_Denoise/tests/test_classical_noise_map.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path helpers so the tests find both scripts/ and mr4raw/src/filters/
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _TESTS_DIR.parent / "scripts"
_MR4RAW_SRC = _TESTS_DIR.parent.parent / "mr4raw" / "src"

# Insert mr4raw/src so `from filters.classical_filters import ...` works
if str(_MR4RAW_SRC) not in sys.path:
    sys.path.insert(0, str(_MR4RAW_SRC))


def _load_script(name: str, path: Path):
    """Dynamically load a script module without running its __main__ block."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Load the script modules lazily at module level so collection errors surface early
_cnm = _load_script(
    "classical_noise_map",
    _SCRIPTS_DIR / "classical_noise_map.py",
)
_brep = _load_script(
    "benchmark_report",
    _SCRIPTS_DIR / "benchmark_report.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Test 1 — mad_sigma_estimate accuracy
# ---------------------------------------------------------------------------

def test_mad_sigma_estimate_accuracy():
    """MAD estimator should recover sigma=0.1 within 30% on a uniform image."""
    from filters.classical_filters import mad_sigma_estimate

    rng = np.random.default_rng(42)
    # Start with a smooth (nearly-uniform) 128x128 image
    clean = np.ones((128, 128), dtype=np.float32) * 0.5
    sigma_true = 0.1
    noisy = clean + rng.normal(0.0, sigma_true, clean.shape).astype(np.float32)

    sigma_est = mad_sigma_estimate(noisy)

    assert sigma_est > 0.0, "Estimated sigma must be positive"
    assert abs(sigma_est - sigma_true) / sigma_true < 0.30, (
        f"MAD estimate {sigma_est:.4f} deviates >30% from true sigma {sigma_true}"
    )


# ---------------------------------------------------------------------------
# Test 2 — wavelet_denoise reduces noise (PSNR improvement)
# ---------------------------------------------------------------------------

def test_wavelet_denoise_reduces_noise():
    """Wavelet denoising should yield higher PSNR than the noisy input."""
    from filters.classical_filters import wavelet_denoise

    rng = np.random.default_rng(0)
    # Synthetic smooth image (2-D sinusoid) so there is signal to recover
    x = np.linspace(0, 1, 128)
    xx, yy = np.meshgrid(x, x)
    clean = (0.5 + 0.4 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)).astype(
        np.float32
    )

    sigma = 0.1
    noisy = np.clip(
        clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32), 0.0, 1.0
    )

    denoised = wavelet_denoise(noisy, sigma=sigma)

    psnr_noisy = _psnr(noisy, clean)
    psnr_denoised = _psnr(denoised, clean)

    assert psnr_denoised > psnr_noisy, (
        f"PSNR did not improve: noisy={psnr_noisy:.2f} dB, "
        f"denoised={psnr_denoised:.2f} dB"
    )


# ---------------------------------------------------------------------------
# Test 3 — generate_report creates HTML file
# ---------------------------------------------------------------------------

def test_benchmark_report_runs(tmp_path: Path):
    """generate_report() must create report.html from a synthetic CSV."""
    import pandas as pd

    # Build a synthetic CSV that matches the benchmark_pretrained.py schema
    rng = np.random.default_rng(7)
    n = 30
    models = ["model_a", "model_b", "noisy_input"]
    records = []
    for model in models:
        for _ in range(n // len(models)):
            records.append(
                {
                    "model": model,
                    "sigma": 0.05,
                    "slice": 0,
                    "psnr": float(rng.uniform(25, 45)),
                    "ssim": float(rng.uniform(0.7, 1.0)),
                    "haarpsi": float(rng.uniform(0.5, 1.0)),
                    "vgg_loss": float(rng.uniform(0.0, 0.5)),
                    "sharpness": float(rng.uniform(0.001, 0.01)),
                }
            )

    csv_path = tmp_path / "benchmark_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    output_dir = tmp_path / "report"
    report_path = _brep.generate_report(csv_path=csv_path, output_dir=output_dir)

    assert report_path.exists(), f"report.html not found at {report_path}"
    assert report_path.suffix == ".html"
    html_content = report_path.read_text(encoding="utf-8")
    # Check that key sections are present
    assert "Leaderboard" in html_content
    assert "PSNR" in html_content
    assert "SSIM" in html_content
    assert "Sharpness" in html_content
    # Figures should be embedded as base64 data URIs
    assert "data:image/png;base64," in html_content
