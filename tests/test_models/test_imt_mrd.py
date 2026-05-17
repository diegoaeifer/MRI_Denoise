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


def _make_wrapper():
    """Create a wrapper with mocked torch.jit.load and pathlib.Path.exists."""
    from imt_mrd_wrapper import ImtMrdWrapper
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 2, 1, 8, 8)
    # Mock both Path.exists and torch.jit.load
    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.jit.load", return_value=mock_model):
            wrapper = ImtMrdWrapper(model_path="fake.pts")
    return wrapper, mock_model


def test_imt_mrd_2d_output_shape():
    """Test 2D input (B, 2, H, W) produces (B, 1, H, W) output."""
    wrapper, mock_model = _make_wrapper()
    x = torch.randn(1, 2, 8, 8)
    out = wrapper(x)
    assert out.shape == (1, 1, 8, 8), f"Expected (1,1,8,8), got {out.shape}"


def test_imt_mrd_3d_output_shape():
    """Test 3D input (B, 2, D, H, W) produces (B, 1, D, H, W) output."""
    wrapper, mock_model = _make_wrapper()
    mock_model.return_value = torch.zeros(1, 2, 4, 8, 8)
    x = torch.randn(1, 2, 4, 8, 8)
    out = wrapper(x)
    assert out.shape == (1, 1, 4, 8, 8), f"Expected (1,1,4,8,8), got {out.shape}"


def test_imt_mrd_output_is_nonneg():
    """Test magnitude output is non-negative."""
    wrapper, mock_model = _make_wrapper()
    # Return non-zero complex components to test magnitude calculation
    mock_model.return_value = torch.randn(1, 2, 1, 8, 8) * 5.0
    x = torch.randn(1, 2, 8, 8)
    out = wrapper(x)
    assert (out >= 0).all(), "Magnitude output should be non-negative"
