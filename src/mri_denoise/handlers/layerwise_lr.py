"""
Layer-wise learning rate decay for fine-tuning pretrained encoders.

When the SwinUNETR encoder unfreezes, shallow (early) encoder layers get
lower LR than the decoder/head to protect pretrained MRI representations.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn


def get_layerwise_lr_groups(
    model: nn.Module,
    base_lr: float,
    decay_factor: float = 0.8,
) -> List[dict]:
    """
    Build optimizer param groups with geometrically decaying LR.

    Parameters are iterated in reverse order (deepest → shallowest encoder).
    Each successive layer gets LR multiplied by `decay_factor`, so the
    shallowest (most sensitive) pretrained layers receive the smallest LR.

    Args:
        model: The network whose parameters to group.
        base_lr: LR for the deepest layer (output head / decoder).
        decay_factor: Per-layer LR multiplier, applied from deep to shallow.

    Returns:
        List of {"params": [...], "lr": float} dicts for torch.optim.
    """
    groups: List[dict] = []
    for i, (_name, param) in enumerate(reversed(list(model.named_parameters()))):
        if param.requires_grad:
            groups.append({"params": [param], "lr": base_lr * (decay_factor**i)})
    return groups
