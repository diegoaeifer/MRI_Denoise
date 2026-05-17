"""Tests for ImT-MRD TorchScript wrapper.

Tests cover:
  - Correct output shape for 2D input (B, 2, H, W) → (B, 1, H, W)
  - Correct output shape for 3D input (B, 2, D, H, W) → (B, 1, D, H, W)
  - Output is non-negative (magnitude constraint)
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

# Add MRI_Denoise root to sys.path so imports work
_MRI_DENOISE_ROOT = Path(__file__).parent.parent.parent
if str(_MRI_DENOISE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MRI_DENOISE_ROOT))

# Add src/models directly to import the wrapper
_MODELS_DIR = _MRI_DENOISE_ROOT / "src" / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))


def _make_wrapper_2d():
    """Create a wrapper with mocked torch.jit.load for 2D tests."""
    from imt_mrd_wrapper import ImtMrdWrapper
    mock_model = MagicMock()
    mock_model.side_effect = lambda x: torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4])
    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.jit.load", return_value=mock_model):
            wrapper = ImtMrdWrapper(model_path="fake.pts")
    return wrapper, mock_model


def _make_wrapper_3d():
    """Create a wrapper with mocked torch.jit.load for 3D tests."""
    from imt_mrd_wrapper import ImtMrdWrapper
    mock_model = MagicMock()
    mock_model.side_effect = lambda x: torch.zeros_like(x)
    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.jit.load", return_value=mock_model):
            wrapper = ImtMrdWrapper(model_path="fake.pts")
    return wrapper, mock_model


def test_imt_mrd_2d_output_shape():
    """Test 2D input (B, 2, H, W) produces (B, 1, H, W) output."""
    wrapper, mock_model = _make_wrapper_2d()
    x = torch.randn(1, 2, 8, 8)
    out = wrapper(x)
    assert out.shape == (1, 1, 8, 8), f"Expected (1,1,8,8), got {out.shape}"


def test_imt_mrd_3d_output_shape():
    """Test 3D input (B, 2, D, H, W) produces (B, 1, D, H, W) output."""
    wrapper, mock_model = _make_wrapper_3d()
    x = torch.randn(1, 2, 4, 8, 8)
    out = wrapper(x)
    assert out.shape == (1, 1, 4, 8, 8), f"Expected (1,1,4,8,8), got {out.shape}"


def test_imt_mrd_output_is_nonneg():
    """Test magnitude output is non-negative."""
    wrapper, mock_model = _make_wrapper_2d()
    # Return non-zero complex components to test magnitude calculation
    mock_model.side_effect = lambda x: torch.randn(x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4]) * 5.0
    x = torch.randn(1, 2, 8, 8)
    out = wrapper(x)
    assert (out >= 0).all(), "Magnitude output should be non-negative"


def test_imt_mrd_raises_on_invalid_dims():
    """Test that invalid input dimensions raise ValueError."""
    wrapper, _ = _make_wrapper_2d()
    with pytest.raises(ValueError, match="Expected 4-D or 5-D"):
        wrapper(torch.randn(1, 2, 8))


def test_imt_mrd_raises_on_wrong_channels():
    """Test that wrong number of channels raises ValueError."""
    wrapper, _ = _make_wrapper_2d()
    with pytest.raises(ValueError, match="Expected 2 input channels"):
        wrapper(torch.randn(1, 3, 8, 8))


def test_factory_registers_imt_mrd():
    """Test that the factory registers 'imt-mrd' model."""
    # Patch torch.jit.load and pathlib.Path.exists before importing factory
    mock_model = MagicMock()
    mock_model.side_effect = lambda x: torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], x.shape[4])

    with patch("torch.jit.load", return_value=mock_model), \
         patch("pathlib.Path.exists", return_value=True):
        # Now import get_model from the src.models package
        import sys
        # Make sure src is in path
        src_path = _MRI_DENOISE_ROOT / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from models import get_model

        config = {
            "common": {"in_channels": 2, "out_channels": 1},
            "imt_mrd": {},
        }
        model = get_model("imt-mrd", config)
    assert model is not None
