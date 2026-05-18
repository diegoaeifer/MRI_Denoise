import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from denoise_benchmark import (
    add_rician_noise,
    solve_l2,
    solve_l1_admm,
    solve_elasticnet_admm,
    solve_tl1_admm,
    solve_wcrr,
    solve_gsd_drunet,
    downscale,
    upscale_lanczos,
    upscale_wcrr,
    compute_metrics,
)

RNG = np.random.default_rng(42)
CLEAN_2D = RNG.uniform(0.2, 0.8, (64, 64))
CLEAN_3D = RNG.uniform(0.2, 0.8, (16, 32, 32))


# --- noise ---
def test_rician_shape_2d():
    assert add_rician_noise(CLEAN_2D, 0.1).shape == CLEAN_2D.shape


def test_rician_shape_3d():
    assert add_rician_noise(CLEAN_3D, 0.1).shape == CLEAN_3D.shape


def test_rician_nonneg():
    assert (add_rician_noise(CLEAN_2D, 0.1) >= 0).all()


# --- denoising reduces MSE ---
def _noisy():
    return add_rician_noise(CLEAN_2D, 0.2)


def test_l2_improves():
    n = _noisy()
    d = solve_l2(n, lam=0.1)
    assert np.mean((d - CLEAN_2D) ** 2) < np.mean((n - CLEAN_2D) ** 2)


def test_l1_improves():
    n = _noisy()
    d = solve_l1_admm(n, lam=0.05)
    assert np.mean((d - CLEAN_2D) ** 2) < np.mean((n - CLEAN_2D) ** 2)


def test_elasticnet_improves():
    n = _noisy()
    d = solve_elasticnet_admm(n, lam1=0.05, lam2=0.05)
    assert np.mean((d - CLEAN_2D) ** 2) < np.mean((n - CLEAN_2D) ** 2)


def test_tl1_improves():
    n = _noisy()
    d = solve_tl1_admm(n, lam=0.05, a=1.0)
    assert np.mean((d - CLEAN_2D) ** 2) < np.mean((n - CLEAN_2D) ** 2)


def test_wcrr_shape():
    assert solve_wcrr(_noisy(), lam=0.1).shape == CLEAN_2D.shape


def test_gsd_drunet_shape():
    assert solve_gsd_drunet(_noisy(), sigma_noise=0.2).shape == CLEAN_2D.shape


# --- SR pipeline ---
def test_downscale_shape():
    assert downscale(CLEAN_2D, 2).shape == (32, 32)


def test_upscale_lanczos():
    assert upscale_lanczos(CLEAN_2D[::2, ::2], CLEAN_2D.shape).shape == CLEAN_2D.shape


def test_upscale_wcrr():
    assert upscale_wcrr(CLEAN_2D[::2, ::2], CLEAN_2D.shape).shape == CLEAN_2D.shape


# --- metrics ---
def test_metrics_perfect():
    m = compute_metrics(CLEAN_2D, CLEAN_2D)
    assert m["psnr"] > 100


def test_metrics_noisy():
    m = compute_metrics(CLEAN_2D, _noisy())
    assert m["psnr"] < 50
