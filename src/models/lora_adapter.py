"""
LoRA adapter for multi-channel MRI denoising.

Extends pretrained 1- or 2-channel models to accept 3-channel input
(magnitude + phase + g-factor map) via:
  1. Input channel expansion: copy existing weights to ch0, zero-init ch1 & ch2
  2. LoRA matrices on linear/conv layers: W = W_base + (B @ A) * scale

Usage
-----
    from models.lora_adapter import expand_input_channels, attach_lora, LoRAWrapper

    # Load pretrained DRUNet
    model = get_model("drunet", config)

    # Expand to 3 channels
    model = expand_input_channels(model, new_in_channels=3)

    # Attach LoRA
    model = attach_lora(model, rank=4, alpha=1.0)

    # Training: freeze base, train only LoRA + new channel weights
    for name, p in model.named_parameters():
        if "lora_" not in name and "new_channels" not in name:
            p.requires_grad = False
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRAConv2d(nn.Module):
    """
    Conv2d with a low-rank additive update: W_eff = W_base + scale * (B @ A)
    where A is (rank, in_ch*k*k) and B is (out_ch, rank).
    """

    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.scale = alpha / rank

        in_features = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
        out_features = conv.out_channels

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # Zero-init B so LoRA output is zero at initialisation (identity init)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base conv output
        base_out = self.conv(x)

        # LoRA delta: reshape weight update to (out_ch, in_ch, kH, kW)
        delta_weight = (self.lora_B @ self.lora_A).view(
            self.conv.weight.shape
        ) * self.scale

        lora_out = F.conv2d(
            x,
            delta_weight,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return base_out + lora_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_first_conv2d(model: nn.Module):
    """
    Return (parent_module, attribute_name, conv_layer) for the first Conv2d
    found in a depth-first traversal of the module tree.

    For nn.Sequential the attribute name is a stringified integer index.
    For named-child modules it is the child name.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Walk the name path to find the direct parent
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                # Support both integer-indexed (Sequential) and named children
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            return parent, parts[-1], module
    raise ValueError("No nn.Conv2d layer found in model.")


def _rsetattr(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """
    Set a (possibly nested) attribute on *model* using a dot-separated path.

    Handles both named attributes and integer-indexed children (nn.Sequential).

    Example
    -------
        _rsetattr(model, "encoder.0.conv", new_conv)
    """
    parts = dotted_name.split(".")
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_input_channels(model: nn.Module, new_in_channels: int = 3) -> nn.Module:
    """
    Expand the first Conv2d layer of model from N channels to new_in_channels.

    Strategy
    --------
    - Copy existing N channels' weights to the first N channels of the new layer.
    - Zero-initialise additional channels so the output is identical to the
      pretrained model when the extra channels are fed zeros.

    Parameters
    ----------
    model : nn.Module
        Pretrained model.  Must have at least one Conv2d layer.
    new_in_channels : int
        Target number of input channels (must be > existing).

    Returns
    -------
    nn.Module
        Model with first conv replaced (in-place modification + return).
    """
    parent, attr_name, old_conv = _find_first_conv2d(model)

    old_in = old_conv.in_channels
    if new_in_channels <= old_in:
        raise ValueError(
            f"new_in_channels ({new_in_channels}) must be greater than "
            f"the existing in_channels ({old_in})."
        )

    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        # Copy pretrained weights for existing channels
        new_conv.weight.data[:, :old_in, :, :] = old_conv.weight.data
        # Zero-initialise new channels (identity at init when extra input = 0)
        new_conv.weight.data[:, old_in:, :, :] = 0.0
        # Copy bias if present
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)

    # Replace in parent
    if attr_name.isdigit():
        parent[int(attr_name)] = new_conv
    else:
        setattr(parent, attr_name, new_conv)

    return model


def attach_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Replace Conv2d layers in model with LoRAConv2d wrappers.

    Parameters
    ----------
    model : nn.Module
    rank : int
        LoRA rank (low = fewer params, high = more expressiveness).
    alpha : float
        LoRA scaling factor.
    target_modules : list of str, optional
        Module name patterns to apply LoRA to.  If None, applies to all Conv2d.
        Example: ["encoder", "decoder"] to skip bottleneck.

    Returns
    -------
    nn.Module
        Model with LoRA wrappers attached.
    """
    # Collect (dotted_name, module) pairs for all Conv2d layers we want to wrap.
    # We snapshot the list before modifying the tree to avoid iteration hazards.
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if target_modules is not None:
            if not any(pat in name for pat in target_modules):
                continue
        targets.append((name, module))

    for name, conv in targets:
        lora_module = LoRAConv2d(conv, rank=rank, alpha=alpha)
        _rsetattr(model, name, lora_module)

    return model


class LoRAWrapper(nn.Module):
    """
    Complete wrapper: expand_input_channels + attach_lora in one step.
    Freezes base weights automatically.
    """

    def __init__(
        self,
        base_model: nn.Module,
        new_in_channels: int = 3,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        model = expand_input_channels(base_model, new_in_channels)
        model = attach_lora(model, rank, alpha)

        # Freeze everything except LoRA params
        for name, param in model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
