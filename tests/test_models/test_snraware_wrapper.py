"""Tests for SNRAwareWrapper model_size and overlap params."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "models"))


def _make_wrapper(model_size="small", overlap=16):
    from snraware_wrapper import SNRAwareWrapper
    mock_model = MagicMock()
    def _fwd(x):
        B, C, T, H, W = x.shape
        return torch.zeros(B, 2, T, H, W)
    mock_model.side_effect = _fwd
    mock_model.parameters = MagicMock(side_effect=lambda: iter([torch.zeros(1)]))
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.glob", return_value=iter(
             [Path(f"weights/SNRAware/{model_size}/snraware_{model_size}_model.pts")])), \
         patch("torch.jit.load", return_value=mock_model):
        w = SNRAwareWrapper(model_path="fake.pts", model_size=model_size,
                            overlap=overlap)
    return w


def test_model_size_stored():
    w = _make_wrapper(model_size="large")
    assert w.model_size == "large"


def test_overlap_stored():
    w = _make_wrapper(overlap=32)
    assert w.overlap == 32


def test_invalid_model_size_raises():
    with pytest.raises(ValueError, match="model_size must be one of"):
        _make_wrapper(model_size="xlarge")


def test_output_shape_with_overlap32():
    w = _make_wrapper(overlap=32)
    x = torch.randn(1, 2, 128, 128)
    out = w(x)
    assert out.shape == (1, 1, 128, 128)
