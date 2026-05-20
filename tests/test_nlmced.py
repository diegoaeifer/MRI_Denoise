"""Tests for the NLmCED 2-D and 3-D filter wrapper."""
import numpy as np
import pytest
import torch

from MRI_Denoise.src.models.nlmced_wrapper import NLmCEDWrapper, nlmced_2d, nlmced_3d

RNG = np.random.default_rng(0)
H, W, D = 48, 48, 16   # small 3-D phantom for fast tests


def _make_phantom_2d(h=H, w=W) -> np.ndarray:
    img = np.full((h, w), 0.2, dtype=np.float64)
    img[h//4:3*h//4, w//4:3*w//4] = 0.7
    img[h//3:2*h//3, w//3:2*w//3] = 0.5
    return img


def _make_phantom_3d(h=H, w=W, d=D) -> np.ndarray:
    vol = np.full((h, w, d), 0.2, dtype=np.float32)
    vol[h//4:3*h//4, w//4:3*w//4, d//4:3*d//4] = 0.7
    vol[h//3:2*h//3, w//3:2*w//3, d//3:2*d//3] = 0.5
    return vol


@pytest.fixture
def clean_img():
    return _make_phantom_2d()


@pytest.fixture
def clean_vol():
    return _make_phantom_3d()


@pytest.fixture
def noisy_tensor_2d():
    img   = _make_phantom_2d().astype(np.float32)
    noisy = np.clip(img + RNG.standard_normal((H, W)).astype(np.float32) * 0.05, 0, 1)
    sigma = np.full((H, W), 0.05, dtype=np.float32)
    return torch.from_numpy(np.stack([noisy, sigma])).unsqueeze(0)  # (1, 2, H, W)


@pytest.fixture
def noisy_tensor_3d():
    vol   = _make_phantom_3d().astype(np.float32)
    noisy = np.clip(vol + RNG.standard_normal((H, W, D)).astype(np.float32) * 0.05, 0, 1)
    sigma = np.full((H, W, D), 0.05, dtype=np.float32)
    # (1, 2, H, W, D)
    return torch.from_numpy(np.stack([noisy, sigma])).unsqueeze(0)


# --------------------------------------------------------------------------- #
# nlmced_2d
# --------------------------------------------------------------------------- #

def test_2d_shape(clean_img):
    assert nlmced_2d(clean_img, iterations=1).shape == (H, W)


def test_2d_range(clean_img):
    noisy = clean_img + RNG.standard_normal((H, W)) * 0.05
    out   = nlmced_2d(noisy, iterations=1)
    assert out.min() > -1.0 and out.max() < 2.0


def test_2d_reduces_noise(clean_img):
    noisy  = clean_img + RNG.standard_normal((H, W)) * 0.08
    out    = nlmced_2d(noisy, iterations=2)
    assert np.mean((out - clean_img)**2) < np.mean((noisy - clean_img)**2)


# --------------------------------------------------------------------------- #
# nlmced_3d
# --------------------------------------------------------------------------- #

def test_3d_shape(clean_vol):
    assert nlmced_3d(clean_vol, iterations=1).shape == (H, W, D)


def test_3d_range(clean_vol):
    noisy = clean_vol + RNG.standard_normal((H, W, D)).astype(np.float32) * 0.05
    out   = nlmced_3d(noisy.astype(np.float64), iterations=1)
    assert out.min() > -1.0 and out.max() < 2.0


def test_3d_reduces_noise(clean_vol):
    noisy = (clean_vol + RNG.standard_normal((H, W, D)).astype(np.float32) * 0.08).astype(np.float64)
    out   = nlmced_3d(noisy, iterations=1)
    assert np.mean((out - clean_vol)**2) < np.mean((noisy - clean_vol)**2)


# --------------------------------------------------------------------------- #
# NLmCEDWrapper — 2-D mode
# --------------------------------------------------------------------------- #

def test_wrapper_2d_shape(noisy_tensor_2d):
    model = NLmCEDWrapper(mode="2d", iterations=1)
    with torch.no_grad():
        out = model(noisy_tensor_2d)
    assert out.shape == (1, 1, H, W)


def test_wrapper_2d_batch(noisy_tensor_2d):
    batch = noisy_tensor_2d.repeat(3, 1, 1, 1)
    model = NLmCEDWrapper(mode="2d", iterations=1)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (3, 1, H, W)


def test_wrapper_2d_auto_dispatch(noisy_tensor_2d):
    model = NLmCEDWrapper(mode="auto", iterations=1)
    with torch.no_grad():
        out = model(noisy_tensor_2d)
    assert out.shape == (1, 1, H, W)


# --------------------------------------------------------------------------- #
# NLmCEDWrapper — 3-D mode
# --------------------------------------------------------------------------- #

def test_wrapper_3d_shape(noisy_tensor_3d):
    model = NLmCEDWrapper(mode="3d", iterations=1)
    with torch.no_grad():
        out = model(noisy_tensor_3d)
    assert out.shape == (1, 1, H, W, D)


def test_wrapper_3d_auto_dispatch(noisy_tensor_3d):
    model = NLmCEDWrapper(mode="auto", iterations=1)
    with torch.no_grad():
        out = model(noisy_tensor_3d)
    assert out.shape == (1, 1, H, W, D)


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #

def test_no_parameters():
    assert sum(p.numel() for p in NLmCEDWrapper().parameters()) == 0


def test_invalid_mode():
    with pytest.raises(ValueError):
        NLmCEDWrapper(mode="banana")


def test_factory_registers_nlmced():
    from MRI_Denoise.src.models.factory import get_model
    cfg = {
        "common": {"in_channels": 2, "out_channels": 1},
        "nlmced": {"iterations": 1, "rho": 0.01, "alpha": 0.01, "num": 1},
    }
    assert isinstance(get_model("nlmced", cfg), NLmCEDWrapper)
