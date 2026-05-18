"""Tests for SNRAwareWrapper use_sigma_as_gmap flag."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "models"))


def _make_wrapper(use_sigma_as_gmap=False, overlap=32):
    from snraware_wrapper import SNRAwareWrapper
    captured = {}

    def _fwd(x):
        captured["input"] = x.clone()
        B, C, T, H, W = x.shape
        return torch.zeros(B, 2, T, H, W)

    mock_model = MagicMock()
    mock_model.side_effect = _fwd
    mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.zeros(1)]))

    with patch("pathlib.Path.exists", return_value=True), \
         patch("torch.jit.load", return_value=mock_model):
        w = SNRAwareWrapper(
            model_path="fake.pts",
            model_size="small",
            overlap=overlap,
            use_sigma_as_gmap=use_sigma_as_gmap,
        )
    return w, captured


def test_default_gmap_is_ones():
    """use_sigma_as_gmap=False -> gmap channel must be all ones."""
    w, captured = _make_wrapper(use_sigma_as_gmap=False)
    sigma_val = 0.10
    x = torch.zeros(1, 2, 64, 64)
    x[:, 1] = sigma_val
    w(x)
    assert "input" in captured, "Model was never called"
    gmap_ch = captured["input"][0, 2]  # channel 2 is g-factor
    assert torch.allclose(gmap_ch, torch.ones_like(gmap_ch)), (
        f"Expected ones for gmap, got min={gmap_ch.min():.4f} max={gmap_ch.max():.4f}"
    )


def test_sigma_gmap_normalizes_to_relative():
    """use_sigma_as_gmap=True -> gmap = sigma_patch / mean(sigma); uniform sigma -> ones."""
    w, captured = _make_wrapper(use_sigma_as_gmap=True)
    sigma_val = 0.10  # uniform sigma -> gmap should be 1 everywhere
    x = torch.zeros(1, 2, 64, 64)
    x[:, 1] = sigma_val
    w(x)
    assert "input" in captured
    gmap_ch = captured["input"][0, 2]
    assert torch.allclose(gmap_ch, torch.ones_like(gmap_ch), atol=1e-4), (
        "Uniform sigma should produce gmap=1 everywhere"
    )


def test_sigma_gmap_nonuniform():
    """Non-uniform sigma -> gmap reflects spatial variation."""
    w, captured = _make_wrapper(use_sigma_as_gmap=True, overlap=0)
    # Left half sigma=0.05, right half sigma=0.15 -> mean=0.10
    x = torch.zeros(1, 2, 64, 64)
    x[:, 1, :, :32] = 0.05
    x[:, 1, :, 32:] = 0.15
    w(x)
    assert "input" in captured
    gmap_ch = captured["input"][0, 2]
    # Left tiles should be < 1, right tiles > 1
    assert gmap_ch[..., :32].mean() < 0.99
    assert gmap_ch[..., 32:].mean() > 1.01


def test_use_sigma_as_gmap_stored():
    """Wrapper stores the flag value for inspection."""
    w, _ = _make_wrapper(use_sigma_as_gmap=True)
    assert w.use_sigma_as_gmap is True
    w2, _ = _make_wrapper(use_sigma_as_gmap=False)
    assert w2.use_sigma_as_gmap is False
