"""Tests for scripts/benchmark_pretrained.py."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
import pytest


def _load():
    spec = importlib.util.spec_from_file_location(
        "benchmark_pretrained",
        Path(__file__).parent.parent / "scripts" / "benchmark_pretrained.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load()


def test_add_rician_noise_shape():
    img = np.random.rand(64, 64).astype(np.float32)
    noisy = bm.add_rician_noise(img, sigma=0.05)
    assert noisy.shape == img.shape
    assert noisy.min() >= 0  # Rician is always positive


def test_add_rician_noise_increases_with_sigma():
    img = np.ones((64, 64), dtype=np.float32) * 0.5
    low = bm.add_rician_noise(img, sigma=0.01)
    high = bm.add_rician_noise(img, sigma=0.20)
    assert np.std(high - img) > np.std(low - img)


def test_psnr_identical():
    t = torch.rand(1, 1, 32, 32)
    assert bm.psnr(t, t) == float("inf")


def test_psnr_decreases_with_noise():
    clean = torch.rand(1, 1, 64, 64)
    noisy_low = clean + 0.01 * torch.randn_like(clean)
    noisy_high = clean + 0.20 * torch.randn_like(clean)
    assert bm.psnr(noisy_low, clean) > bm.psnr(noisy_high, clean)


def test_sharpness_blurred_lower(tmp_path):
    from scipy.ndimage import gaussian_filter
    sharp = torch.from_numpy(np.random.rand(64, 64).astype(np.float32))
    blurred_arr = gaussian_filter(sharp.numpy(), sigma=3)
    blurred = torch.from_numpy(blurred_arr)
    assert bm.sharpness(sharp) > bm.sharpness(blurred)


def test_make_sigma_map():
    device = torch.device("cpu")
    m = bm.make_sigma_map((1, 1, 32, 32), 0.05, device)
    assert m.shape == (1, 1, 32, 32)
    assert torch.allclose(m, torch.full_like(m, 0.05))


def test_benchmark_runs_with_dummy_model(tmp_path):
    """End-to-end: benchmark with a trivial identity model on synthetic data."""
    pydicom = pytest.importorskip("pydicom")
    # pydicom 3.x renamed get_testfiles_name → get_testdata_file
    try:
        from pydicom.data import get_testfiles_name
        dcm_path = get_testfiles_name("CT_small.dcm")
    except ImportError:
        from pydicom.data import get_testdata_file
        dcm_path = get_testdata_file("CT_small.dcm")
    import shutil
    dcm_dir = tmp_path / "dicom"
    dcm_dir.mkdir()
    shutil.copy(dcm_path, dcm_dir / "test.dcm")

    # Monkey-patch models to avoid needing real weights
    identity = torch.nn.Identity()

    class _IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x[:, :1, :, :]  # return just the image channel

    orig_load = bm.load_model

    def _fake_load(name, device):
        return _IdentityModel().eval()

    bm.load_model = _fake_load
    try:
        df = bm.benchmark(
            data_dir=dcm_dir,
            output_dir=tmp_path / "out",
            noise_levels=[0.05],
            model_names=["identity_test"],
            max_slices=2,
            device=torch.device("cpu"),
        )
        assert len(df) > 0
        assert "psnr" in df.columns
        assert "ssim" in df.columns
    finally:
        bm.load_model = orig_load
