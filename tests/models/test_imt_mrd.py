import pytest
import torch
from unittest.mock import patch, MagicMock


def _make_wrapper(tmp_path=None):
    from MRI_Denoise.src.models.imt_mrd_wrapper import ImtMrdWrapper

    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 2, 1, 8, 8)

    with patch("torch.jit.load", return_value=mock_model):
        wrapper = ImtMrdWrapper(model_path="fake.pts")
    return wrapper, mock_model


def test_imt_mrd_2d_output_shape():
    wrapper, mock_model = _make_wrapper()
    x = torch.randn(1, 2, 8, 8)
    out = wrapper(x)
    assert out.shape == (1, 1, 8, 8), f"Expected (1,1,8,8), got {out.shape}"


def test_imt_mrd_3d_output_shape():
    wrapper, mock_model = _make_wrapper()
    mock_model.return_value = torch.zeros(1, 2, 4, 8, 8)
    x = torch.randn(1, 2, 4, 8, 8)
    out = wrapper(x)
    assert out.shape == (1, 1, 4, 8, 8), f"Expected (1,1,4,8,8), got {out.shape}"


def test_imt_mrd_output_is_nonneg():
    wrapper, mock_model = _make_wrapper()
    mock_model.return_value = torch.randn(1, 2, 1, 8, 8)
    x = torch.randn(1, 2, 8, 8)
    out = wrapper(x)
    assert (out >= 0).all()
