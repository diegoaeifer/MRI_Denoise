import sys
from unittest.mock import MagicMock
import pytest

# Helper to mock dependencies before importing src.models.factory
def setup_mocks():
    mock_torch = MagicMock()
    mock_nn = MagicMock()
    mock_nn_functional = MagicMock()

    # sys.modules['torch'] = mock_torch
    # sys.modules['torch.nn'] = mock_nn
    # sys.modules['torch.nn.functional'] = mock_nn_functional
    # sys.modules['torchio'] = MagicMock()
    # sys.modules['monai'] = MagicMock()
    sys.modules['einops'] = MagicMock()

setup_mocks()

from src.models.factory import get_model

@pytest.fixture
def model_config():
    return {
        'common': {
            'in_channels': 2,
            'out_channels': 1
        },
        'drunet': {
            'base_channels': 32
        },
        'nafnet': {
            'width': 16,
            'enc_blk_nums': [1, 1, 1, 1],
            'middle_blk_num': 1,
            'dec_blk_nums': [1, 1, 1, 1]
        },
        'nafnet_small': {
            'width': 12,
            'enc_blk_nums': [2, 2, 2, 2],
            'middle_blk_num': 2,
            'dec_blk_nums': [2, 2, 2, 2]
        },
        'nafnet_medium': {
            'width': 24,
            'enc_blk_nums': [3, 3, 3, 3],
            'middle_blk_num': 3,
            'dec_blk_nums': [3, 3, 3, 3]
        },
        'nafnet_large': {
            'width': 32,
            'enc_blk_nums': [4, 4, 4, 4],
            'middle_blk_num': 4,
            'dec_blk_nums': [4, 4, 4, 4]
        },
        'scunet': {
            'config': 'A'
        },
        'unet': {
            'bilinear': True
        }
    }

def test_get_drunet(model_config, monkeypatch):
    mock_class = MagicMock()
    monkeypatch.setattr("src.models.factory.DRUNet", mock_class)
    model = get_model('drunet', model_config)
    mock_class.assert_called_once_with(
        in_channels=2,
        out_channels=1,
        base_channels=32
    )
    assert model == mock_class.return_value

def test_get_nafnet(model_config, monkeypatch):
    mock_class = MagicMock()
    monkeypatch.setattr("src.models.factory.NAFNet", mock_class)

    # Test base nafnet
    get_model('nafnet', model_config)
    mock_class.assert_called_with(
        img_channel=2,
        width=16,
        enc_blk_nums=[1, 1, 1, 1],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1]
    )

    # Test nafnet_small
    get_model('nafnet_small', model_config)
    mock_class.assert_called_with(
        img_channel=2,
        width=12,
        enc_blk_nums=[2, 2, 2, 2],
        middle_blk_num=2,
        dec_blk_nums=[2, 2, 2, 2]
    )

def test_get_scunet(model_config, monkeypatch):
    mock_class = MagicMock()
    monkeypatch.setattr("src.models.factory.SCUNet", mock_class)
    model = get_model('scunet', model_config)
    mock_class.assert_called_once_with(
        in_channels=2,
        out_channels=1,
        config='A'
    )
    assert model == mock_class.return_value

def test_get_unet(model_config, monkeypatch):
    mock_class = MagicMock()
    monkeypatch.setattr("src.models.factory.UNet", mock_class)
    model = get_model('unet', model_config)
    mock_class.assert_called_once_with(
        n_channels=2,
        n_classes=1,
        bilinear=True
    )
    assert model == mock_class.return_value

def test_get_model_case_insensitive(model_config, monkeypatch):
    mock_class = MagicMock()
    monkeypatch.setattr("src.models.factory.DRUNet", mock_class)
    model = get_model('DRUNet', model_config)
    assert model == mock_class.return_value

def test_get_model_invalid_name(model_config):
    with pytest.raises(ValueError, match="Model unknown_model not implemented."):
        get_model('unknown_model', model_config)
