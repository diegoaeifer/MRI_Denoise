"""Tests for CDLNet net_state_dict key fix and K=30 architecture."""
import sys
from pathlib import Path
import pytest
import torch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "models"))
sys.path.insert(0, str(_ROOT / "FMImaging_MRI_Denoise" / "CDLNET"))

try:
    from unittest.mock import MagicMock
    sys.modules.setdefault("torchvision", MagicMock())
    sys.modules.setdefault("torchvision.transforms", MagicMock())
    sys.modules.setdefault("torchvision.transforms.functional", MagicMock())
except Exception:
    pass


def test_net_state_dict_key_loads(tmp_path):
    """weights_path with 'net_state_dict' top key must load correctly."""
    from cdlnet_wrapper import CDLNetWrapper
    w = CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=False, init=False)
    ckpt = tmp_path / "cdlnet_s2030_style.ckpt"
    torch.save({"net_state_dict": w.net.state_dict(), "epoch": 42}, str(ckpt))
    w2 = CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=False,
                       init=False, weights_path=str(ckpt))
    for k in w.net.state_dict():
        assert torch.allclose(w.net.state_dict()[k], w2.net.state_dict()[k]), (
            f"Parameter mismatch at {k}"
        )


def test_state_dict_key_still_works(tmp_path):
    """Existing checkpoints saved with 'state_dict' key must still load."""
    from cdlnet_wrapper import CDLNetWrapper
    w = CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=False, init=False)
    ckpt = tmp_path / "old_style.pth"
    torch.save({"state_dict": w.net.state_dict()}, str(ckpt))
    w2 = CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=False,
                       init=False, weights_path=str(ckpt))
    for k in w.net.state_dict():
        assert torch.allclose(w.net.state_dict()[k], w2.net.state_dict()[k])


def test_missing_weights_path_raises():
    from cdlnet_wrapper import CDLNetWrapper
    with pytest.raises(FileNotFoundError, match="CDLNet weights not found"):
        CDLNetWrapper(K=3, M=8, P=3, s=1, adaptive=False, init=False,
                      weights_path="/nonexistent/path/net.ckpt")
