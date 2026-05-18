"""Tests for FFDNetKAIRWrapper."""
import sys
from pathlib import Path
import pytest
import torch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "models"))
sys.path.insert(0, str(_ROOT / "FMImaging_MRI_Denoise" / "KAIR"))


def _kair_available() -> bool:
    return (_ROOT / "FMImaging_MRI_Denoise" / "KAIR" / "models" / "network_ffdnet.py").exists()


pytestmark = pytest.mark.skipif(
    not _kair_available(),
    reason="KAIR repo not cloned to FMImaging_MRI_Denoise/KAIR"
)


def _model():
    from ffdnet_kair_wrapper import FFDNetKAIRWrapper
    return FFDNetKAIRWrapper(in_nc=1, out_nc=1, nc=64, nb=15,
                             act_mode="BR", weights_path=None)


def test_instantiation():
    m = _model()
    assert m is not None


@pytest.mark.parametrize("b,h,w", [(1, 32, 32), (2, 64, 64), (1, 48, 64)])
def test_output_shape(b, h, w):
    m = _model().eval()
    x = torch.randn(b, 2, h, w)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (b, 1, h, w), f"Expected ({b},1,{h},{w}), got {out.shape}"


def test_no_nan_or_inf():
    m = _model().eval()
    x = torch.randn(1, 2, 32, 32)
    with torch.no_grad():
        out = m(x)
    assert not out.isnan().any(), "Output contains NaN"
    assert not out.isinf().any(), "Output contains Inf"


def test_gradient_flow():
    m = _model().train()
    x = torch.randn(1, 2, 32, 32).requires_grad_(True)
    out = m(x)
    out.mean().backward()
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_weight_loading(tmp_path):
    """weights_path loads state_dict correctly."""
    from ffdnet_kair_wrapper import FFDNetKAIRWrapper
    w = FFDNetKAIRWrapper(in_nc=1, out_nc=1, nc=64, nb=15,
                          act_mode="BR", weights_path=None)
    ckpt = tmp_path / "ffdnet_test.pth"
    torch.save(w.net.state_dict(), str(ckpt))
    w2 = FFDNetKAIRWrapper(in_nc=1, out_nc=1, nc=64, nb=15,
                           act_mode="BR", weights_path=str(ckpt))
    for k in w.net.state_dict():
        assert torch.allclose(w.net.state_dict()[k], w2.net.state_dict()[k])
