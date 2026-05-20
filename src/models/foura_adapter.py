"""
FouRA (Fourier Low-Rank Adaptation) adapter for Conv2d layers.

NeurIPS 2024: Instead of standard LoRA  out = W(x) + scale*B(A(x)),
FouRA operates in the DCT frequency domain:
    out = W(x) + scale * IDCT2(B @ A @ DCT2(x))

The DCT2/IDCT2 transform encourages learning low-frequency structure
which is well-suited for MRI denoising tasks.

Usage
-----
    from src.models.foura_adapter import FouRAConv2d, attach_foura, FouRAWrapper

    model = NAFNet(img_channel=2, width=64)
    wrapped = FouRAWrapper(model, rank=16, alpha=32.0)
    # Only foura_A and foura_B are trainable
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch_dct


def _dct2(x: torch.Tensor) -> torch.Tensor:
    """Apply 2-D DCT (ortho-normalised) over the spatial dims of a 4-D tensor."""
    return torch_dct.dct_2d(x, norm="ortho")


def _idct2(x: torch.Tensor) -> torch.Tensor:
    """Apply 2-D inverse DCT (ortho-normalised) over the spatial dims of a 4-D tensor."""
    return torch_dct.idct_2d(x, norm="ortho")


class FouRAConv2d(nn.Module):
    """Conv2d + FouRA adapter.

    Forward pass:
        out = conv(x) + scale * IDCT2(B @ A @ DCT2(x))

    where A is (rank, in_channels) and B is (out_channels, rank).
    B is zero-initialised so FouRA is identity (adds nothing) at init.
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.conv = conv
        # Freeze the base conv weights
        for p in self.conv.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        # A: channel down-projection in freq domain
        self.foura_A = nn.Parameter(torch.empty(rank, conv.in_channels))
        # B: zero-init so adapter contributes nothing at initialisation
        self.foura_B = nn.Parameter(torch.zeros(conv.out_channels, rank))
        nn.init.kaiming_uniform_(self.foura_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base convolution output
        base = self.conv(x)

        # --- FouRA branch ---
        # x: (B, C_in, H, W)
        x_freq = _dct2(x)                                      # (B, C_in, H, W) in freq domain

        B_sz, C, H, W = x_freq.shape
        # Flatten spatial dims for matrix multiply: (B*H*W, C_in)
        x_flat = x_freq.permute(0, 2, 3, 1).reshape(B_sz * H * W, C)

        # Down-project with A: (B*H*W, rank)
        low = x_flat @ self.foura_A.T
        # Up-project with B: (B*H*W, C_out)
        high = low @ self.foura_B.T

        # Reshape back to (B, C_out, H, W)
        adapter_freq = high.reshape(B_sz, H, W, -1).permute(0, 3, 1, 2)

        # Back to spatial domain
        adapter_spatial = _idct2(adapter_freq)

        return base + self.scale * adapter_spatial

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, scale={self.scale:.4f}, "
            f"in={self.conv.in_channels}, out={self.conv.out_channels}"
        )


def attach_foura(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    """Replace Conv2d layers (except 1×1) with FouRAConv2d wrappers.

    Parameters
    ----------
    model : nn.Module
    rank : int
        FouRA rank.
    alpha : float
        Scaling factor (scale = alpha / rank).
    target_modules : list of str, optional
        Name patterns to limit which Conv2d layers are wrapped.
        If None, wraps all non-1×1 Conv2d layers.

    Returns
    -------
    nn.Module
        Model with FouRA adapters attached (modified in-place, also returned).
    """
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        # Skip 1×1 convolutions (point-wise, no spatial structure to exploit)
        if module.kernel_size == (1, 1):
            continue
        # Skip stride>1 convolutions (adapter output H,W would mismatch base conv output)
        if module.stride != (1, 1):
            continue
        if target_modules and not any(t in name for t in target_modules):
            continue
        targets.append(name)

    for name in targets:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old_conv = getattr(parent, parts[-1])
        setattr(parent, parts[-1], FouRAConv2d(old_conv, rank=rank, alpha=alpha))

    return model


class FouRAWrapper(nn.Module):
    """Freeze backbone, then attach FouRA adapters to all non-1×1 Conv2d layers.

    Only the foura_A and foura_B parameters (one pair per wrapped conv) are
    set to requires_grad=True; everything else is frozen.
    """

    def __init__(
        self,
        base_model: nn.Module,
        rank: int = 16,
        alpha: float = 1.0,
        target_modules: Optional[list[str]] = None,
    ):
        super().__init__()
        # Freeze all base parameters first
        for p in base_model.parameters():
            p.requires_grad = False
        # Attach adapters (FouRAConv2d freezes the wrapped conv internally too)
        self.model = attach_foura(base_model, rank=rank, alpha=alpha, target_modules=target_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_stats(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "ratio": trainable / total if total > 0 else 0.0,
        }
