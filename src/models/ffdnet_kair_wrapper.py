"""Wrapper for FFDNet from KAIR (cszn/KAIR).

Input convention: (B, 2, H, W) — ch0=noisy image in [0,1], ch1=sigma map in [0,1].
KAIR FFDNet expects images in [0,1] and sigma in [0,1] (noise_level/255).
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_KAIR_REPO = Path(__file__).parent.parent.parent / "FMImaging_MRI_Denoise" / "KAIR"


def _import_ffdnet():
    """Import FFDNet by absolute path, avoiding name conflict with src/models package.

    network_ffdnet.py uses `import models.basicblock as B` which would normally
    resolve to our src/models package. We temporarily replace sys.modules['models']
    with a fake module that has basicblock, execute network_ffdnet.py, then restore.
    """
    import importlib.util
    import types

    ff_key = "kair.models.network_ffdnet"
    if ff_key in sys.modules:
        ffdnet_mod = sys.modules[ff_key]
        if hasattr(ffdnet_mod, "FFDNet"):
            return ffdnet_mod.FFDNet
        del sys.modules[ff_key]  # Remove broken cached entry

    bb_path = _KAIR_REPO / "models" / "basicblock.py"
    ff_path = _KAIR_REPO / "models" / "network_ffdnet.py"
    if not ff_path.exists():
        raise ImportError(
            "KAIR FFDNet not found. "
            f"Clone to {_KAIR_REPO}: git clone https://github.com/cszn/KAIR {_KAIR_REPO}"
        )

    # Load basicblock by file path under a namespaced key
    bb_key = "kair.models.basicblock"
    if bb_key not in sys.modules:
        spec = importlib.util.spec_from_file_location(bb_key, str(bb_path))
        bb_mod = importlib.util.module_from_spec(spec)
        sys.modules[bb_key] = bb_mod
        spec.loader.exec_module(bb_mod)
    bb_mod = sys.modules[bb_key]

    # Temporarily replace sys.modules['models'] with a fake that has basicblock,
    # so `import models.basicblock as B` in network_ffdnet.py doesn't resolve to src/models
    _saved_models = sys.modules.get("models")
    _saved_models_bb = sys.modules.get("models.basicblock")
    fake_models = types.ModuleType("models")
    fake_models.basicblock = bb_mod
    sys.modules["models"] = fake_models
    sys.modules["models.basicblock"] = bb_mod

    try:
        spec = importlib.util.spec_from_file_location(ff_key, str(ff_path))
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        sys.modules[ff_key] = ff_mod
        return ff_mod.FFDNet
    except Exception as exc:
        raise ImportError(
            f"KAIR FFDNet import failed: {exc}. "
            f"Check {_KAIR_REPO}"
        ) from exc
    finally:
        # Restore whatever was in sys.modules['models'] before our patch
        if _saved_models is not None:
            sys.modules["models"] = _saved_models
        elif "models" in sys.modules:
            del sys.modules["models"]
        if _saved_models_bb is not None:
            sys.modules["models.basicblock"] = _saved_models_bb
        elif "models.basicblock" in sys.modules:
            del sys.modules["models.basicblock"]


class FFDNetKAIRWrapper(nn.Module):
    """Factory-standard wrapper for FFDNet (KAIR).

    Parameters
    ----------
    in_nc : input channels (1 for grayscale).
    out_nc : output channels (1 for grayscale).
    nc : number of feature maps (64 recommended).
    nb : number of conv blocks (15 recommended).
    act_mode : activation mode ('R'=ReLU, 'BR'=BN+ReLU).
    weights_path : path to `ffdnet_gray.pth` or None for random init.
    """

    def __init__(
        self,
        in_channels: int = 2,   # factory convention (not used in network init)
        in_nc: int = 1,
        out_nc: int = 1,
        nc: int = 64,
        nb: int = 15,
        act_mode: str = "R",
        weights_path: str | None = None,
    ):
        super().__init__()
        FFDNet = _import_ffdnet()
        self.net = FFDNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb,
                          act_mode=act_mode)
        if weights_path is not None:
            p = Path(weights_path)
            if not p.exists():
                raise FileNotFoundError(f"FFDNet-KAIR weights not found: {p}")
            raw = torch.load(str(p), map_location="cpu", weights_only=False)
            # KAIR checkpoints may use 'params' wrapper key or be raw state dicts
            state = raw.get("params", raw) if isinstance(raw, dict) else raw
            self.net.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) -> (B, 1, H, W)

        Image and sigma are both expected in [0,1]; KAIR FFDNet uses the same range.
        """
        img = x[:, :1, :, :]
        sigma_map = x[:, 1:2, :, :]
        sigma_val = sigma_map.mean().item()
        sigma_tensor = torch.full(
            (img.shape[0], 1, 1, 1), sigma_val,
            dtype=img.dtype, device=img.device,
        )
        out = self.net(img, sigma_tensor)
        return out.clamp(0.0, 1.0)
