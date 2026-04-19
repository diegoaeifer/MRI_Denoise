import pytest
import torch
import unittest.mock as mock
from src.models.factory import get_model

@pytest.fixture(autouse=True)
def mock_dependencies():
    # Provide necessary setup if any
    pass

class DummyModel:
    def __init__(self, ret):
        self.ret = ret
        self.parameters = mock.MagicMock(return_value=[mock.MagicMock()])
    def __call__(self, x):
        return self.ret

def test_snraware_factory_creation():
    config = {
        'common': {
            'in_channels': 2,
            'out_channels': 1
        },
        'snraware': {
            'pretrained': 'src/models/snraware_small_model.pts',
            'freeze': True
        }
    }

    with mock.patch('torch.jit.load') as mock_load:
        class DummyModel:
            def __init__(self, ret):
                self.ret = ret
                self.parameters = mock.MagicMock(return_value=[mock.MagicMock()])
            def __call__(self, x):
                return self.ret
        mock_model = DummyModel(torch.ones(1, 2, 1, 32, 32))
        mock_load.return_value = mock_model

        model = get_model('snraware', config)

        # Test freeze works: since we mock parameters we can test the mock
        mock_model.parameters()[0].requires_grad_.assert_called_with(False)
        assert hasattr(model, 'model')

def test_snraware_factory_forward_2d():
    config = {
        'common': {
            'in_channels': 2,
            'out_channels': 1
        },
        'snraware': {
            'pretrained': 'src/models/snraware_small_model.pts'
        }
    }
    with mock.patch('torch.jit.load') as mock_load:
        mock_model = DummyModel(torch.ones(1, 2, 1, 32, 32))
        mock_load.return_value = mock_model

        model = get_model('snraware', config)

        x = torch.randn(1, 2, 32, 32)
        out = model(x)

        assert out.shape == (1, 1, 32, 32)

def test_snraware_factory_forward_3d():
    config = {
        'common': {
            'in_channels': 2,
            'out_channels': 1
        },
        'snraware': {
            'pretrained': 'src/models/snraware_small_model.pts'
        }
    }
    with mock.patch('torch.jit.load') as mock_load:
        class DummyModel:
            def __init__(self, ret):
                self.ret = ret
                self.parameters = mock.MagicMock(return_value=[mock.MagicMock()])
            def __call__(self, x):
                return self.ret
        mock_model = DummyModel(torch.ones(1, 2, 16, 32, 32))
        mock_load.return_value = mock_model

        model = get_model('snraware', config)

        x = torch.randn(1, 2, 16, 32, 32)
        out = model(x)

        assert out.shape == (1, 1, 16, 32, 32)
