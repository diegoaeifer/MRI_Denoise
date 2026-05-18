"""Wrapper for CDLNET (nikopj/CDLNET-OJSP).

Input convention: (B, 2, H, W) — ch0=noisy image, ch1=sigma map in [0,1].
CDLNet internally uses sigma/255, so we pass sigma_map.mean()*255.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_CDLNET_REPO = Path(__file__).parent.parent.parent / "FMImaging_MRI_Denoise" / "CDLNET"


def _import_cdlnet():
    repo_str = str(_CDLNET_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        from model.net import CDLNet  # noqa: PLC0415
        return CDLNet
    except ImportError as exc:
        raise ImportError(
            "CDLNet could not be imported. "
            f"Clone to {_CDLNET_REPO}: "
            "git clone https://github.com/nikopj/CDLNET-OJSP "
            f"{_CDLNET_REPO}"
        ) from exc


class CDLNetWrapper(nn.Module):
    """Factory-standard wrapper for CDLNet.

    Parameters
    ----------
    K, M, P, s : CDLNet architecture (unrolling depth, filters, patch, stride).
    adaptive : use noise-adaptive thresholds (requires sigma hint).
    init : run power-method init (set False for pretrained checkpoints).
    weights_path : .pth/.pt checkpoint to load. Keys 'state_dict' or 'model'
        are unwrapped automatically. Ignored if None.
    """

    def __init__(
        self,
        in_channels: int = 2,
        K: int = 30,
        M: int = 169,
        P: int = 7,
        s: int = 2,
        adaptive: bool = True,
        init: bool = False,
        weights_path: str | None = None,
    ):
        super().__init__()
        CDLNet = _import_cdlnet()
        self.net = CDLNet(K=K, M=M, P=P, s=s, C=1,
                          adaptive=adaptive, init=init)
        self.adaptive = adaptive

        if weights_path is not None:
            p = Path(weights_path)
            if not p.exists():
                raise FileNotFoundError(f"CDLNet weights not found: {p}")
            raw = torch.load(str(p), map_location="cpu", weights_only=False)
            state = (
                raw.get("net_state_dict")
                or raw.get("state_dict")
                or raw.get("model")
                or raw
            )
            self.net.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) -> (B, 1, H, W)"""
        img = x[:, :1, :, :]
        sigma_map = x[:, 1:2, :, :]
        # CDLNet divides sigma by 255 internally; our sigma map is in [0,1]
        sigma = sigma_map.mean().item() * 255.0 if self.adaptive else None
        xhat, _z = self.net(img, sigma=sigma)
        return xhat
