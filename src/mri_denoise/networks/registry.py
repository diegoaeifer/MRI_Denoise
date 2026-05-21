"""
Network registry — replaces the 270-line if/elif factory.py.

Adding a new model is one line in REGISTRY. All kwargs come from config.
Uses MONAI native models where available; keeps custom models that
have no MONAI equivalent.

All custom models MUST accept: in_channels, out_channels, spatial_dims
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# 2-channel adapter (image + sigma_map → 1-channel for single-channel backbones)
# ---------------------------------------------------------------------------- #


class TwoChannelAdapter(nn.Module):
    """
    Fuses (image, sigma_map) 2-channel input into 1-channel for pretrained
    single-channel backbones. Learnable; spatial_dims-aware.
    """

    def __init__(self, in_channels: int = 2, spatial_dims: int = 2) -> None:
        super().__init__()
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.fuse = Conv(in_channels, 1, kernel_size=1)
        # near-identity init: mostly pass through image channel
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        with torch.no_grad():
            self.fuse.weight[0, 0] = 0.9  # image channel
            if in_channels > 1:
                self.fuse.weight[0, 1] = 0.1  # sigma channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(x)


# ---------------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------------- #

def _lazy_import(module: str, cls: str):
    """Lazily import a class to avoid import errors for optional deps."""
    def factory(**kwargs):
        import importlib
        mod = importlib.import_module(module)
        return getattr(mod, cls)(**kwargs)
    factory.__name__ = cls
    return factory


# MONAI native models
def _build_basic_unet(**kwargs):
    from monai.networks.nets import BasicUNet
    return BasicUNet(**kwargs)


def _build_attention_unet(**kwargs):
    from monai.networks.nets import AttentionUnet
    return AttentionUnet(**kwargs)


def _build_dynunet(**kwargs):
    from monai.networks.nets import DynUNet
    return DynUNet(**kwargs)


def _build_swinunetr(**kwargs):
    from monai.networks.nets import SwinUNETR
    return SwinUNETR(**kwargs)


def _build_swinunetr_denoising(**kwargs):
    from .swinunetr_denoising import SwinUNETRDenoising
    return SwinUNETRDenoising(**kwargs)


# Custom models (no MONAI equivalent)
def _build_nafnet(**kwargs):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from src.models.nafnet import NAFNet
    # Map MONAI-style kwargs to NAFNet kwargs
    in_ch = kwargs.pop("in_channels", 2)
    kwargs.pop("out_channels", 1)
    kwargs.pop("spatial_dims", 2)
    return NAFNet(img_channel=in_ch, **kwargs)


def _build_drunet(**kwargs):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from src.models.drunet import DRUNet
    kwargs.pop("spatial_dims", 2)
    return DRUNet(**kwargs)


def _build_ricianet3d(**kwargs):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from src.models.rician_net3d import RicianNet3D
    kwargs.pop("spatial_dims", 3)
    return RicianNet3D(**kwargs)


def _build_snraware(**kwargs):
    from .snraware_adapter import SNRAwareDenoisingAdapter
    return SNRAwareDenoisingAdapter(**kwargs)


REGISTRY: Dict[str, Any] = {
    # MONAI native (2D/3D, use for standard tasks)
    "basic_unet": _build_basic_unet,
    "attention_unet": _build_attention_unet,
    "dynunet": _build_dynunet,
    "swinunetr": _build_swinunetr,
    # SwinUNETR fine-tuned for denoising (Step 2 of migration plan)
    "swinunetr_denoising": _build_swinunetr_denoising,
    # Custom architectures (not in MONAI)
    "nafnet": _build_nafnet,
    "drunet": _build_drunet,
    "ricianet3d": _build_ricianet3d,  # 3D native
    # SNRAware (Microsoft) — requires optional snraware package
    "snraware": _build_snraware,
}


def get_network(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name: model name (key in REGISTRY)
        **kwargs: model hyperparams (in_channels, out_channels, spatial_dims, etc.)

    Returns:
        nn.Module instance

    Raises:
        KeyError: if name not in REGISTRY
    """
    name = name.lower()
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown network: '{name}'. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    model = REGISTRY[name](**kwargs)
    logger.info(
        f"Built model '{name}' | "
        f"params: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model
