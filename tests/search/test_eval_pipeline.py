# MRI_Denoise/tests/search/test_eval_pipeline.py
"""Integration tests for eval_pipeline.

Uses synthetic images and a CPU stub (identity model) so tests run without
downloading pretrained weights or using GPU.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from search.eval_pipeline import (
    add_rician_noise,
    build_input_tensor,
    run_trial_cpu_stub,
)


def test_add_rician_noise_changes_image():
    clean = np.ones((32, 32), dtype=np.float32) * 0.5
    rng = np.random.default_rng(0)
    noisy = add_rician_noise(clean, sigma=0.1, rng=rng)
    assert noisy.shape == clean.shape
    assert not np.array_equal(noisy, clean)
    assert noisy.min() >= 0.0


def test_rician_noise_sigma_0_returns_near_clean():
    clean = np.random.rand(32, 32).astype(np.float32)
    rng = np.random.default_rng(1)
    noisy = add_rician_noise(clean, sigma=1e-6, rng=rng)
    np.testing.assert_allclose(noisy, clean, atol=1e-4)


def test_build_input_tensor_shape_2d():
    import torch
    noisy = np.random.rand(64, 64).astype(np.float32)
    t = build_input_tensor(noisy, sigma=0.1, gmap_strategy="uniform", mode="2d")
    assert t.shape == (1, 2, 64, 64)
    assert t.dtype == torch.float32


def test_build_input_tensor_shape_3d():
    import torch
    noisy = np.random.rand(64, 64, 16).astype(np.float32)
    t = build_input_tensor(noisy, sigma=0.1, gmap_strategy="uniform", mode="3d")
    # (B=1, C=2, D=16, H=64, W=64)
    assert t.shape == (1, 2, 16, 64, 64)


def test_run_trial_cpu_stub_returns_psnr_and_ssim():
    result = run_trial_cpu_stub(sigma=0.1, gmap_strategy="uniform",
                                unsharp_cfg={"name": "none"},
                                dither_cfg={"name": "none"})
    assert "psnr" in result
    assert "ssim" in result
    assert result["psnr"] > 0.0
    assert 0.0 < result["ssim"] <= 1.0
    assert result.get("error") is None
