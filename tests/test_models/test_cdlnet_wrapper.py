"""Tests for CDLNetWrapper sigma scaling and weight loading."""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "models"))
sys.path.insert(0, str(_ROOT / "FMImaging_MRI_Denoise" / "CDLNET"))

# CDLNET/utils.py imports torchvision which is incompatible with the test venv's
# CPU torch build. Pre-mock it so the CDLNET model module can be imported.
_tv_mock = MagicMock()
sys.modules.setdefault('torchvision', _tv_mock)
sys.modules.setdefault('torchvision.transforms', _tv_mock)
sys.modules.setdefault('torchvision.transforms.functional', _tv_mock)


def _make_wrapper(adaptive=True):
    from cdlnet_wrapper import CDLNetWrapper
    return CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=adaptive, init=False)


def test_sigma_scaling_reaches_net():
    """adaptive=True must pass sigma*255 to CDLNet, not raw sigma."""
    wrapper = _make_wrapper(adaptive=True)
    captured = {}
    original = wrapper.net.forward
    def patched(y, sigma=None, mask=1):
        captured['sigma'] = sigma
        return original(y, sigma=sigma, mask=mask)
    wrapper.net.forward = patched
    x = torch.zeros(1, 2, 16, 16)
    x[:, 1, :, :] = 0.10  # sigma map = 0.10 normalized
    wrapper(x)
    assert captured['sigma'] is not None
    assert captured['sigma'] > 1.0, (
        f"sigma passed to CDLNet = {captured['sigma']:.4f}; "
        "expected ~25.5 (0.10 * 255)"
    )


def test_non_adaptive_passes_none():
    wrapper = _make_wrapper(adaptive=False)
    captured = {}
    original = wrapper.net.forward
    def patched(y, sigma=None, mask=1):
        captured['sigma'] = sigma
        return original(y, sigma=sigma, mask=mask)
    wrapper.net.forward = patched
    x = torch.zeros(1, 2, 16, 16)
    wrapper(x)
    assert captured['sigma'] is None


def test_weight_loading(tmp_path):
    """weights_path loads state_dict from .pth checkpoint."""
    from cdlnet_wrapper import CDLNetWrapper
    w = CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=False, init=False)
    ckpt = tmp_path / "cdlnet_test.pth"
    torch.save({"state_dict": w.net.state_dict()}, str(ckpt))
    w2 = CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=False,
                       init=False, weights_path=str(ckpt))
    for k in w.net.state_dict():
        assert torch.allclose(w.net.state_dict()[k], w2.net.state_dict()[k])
