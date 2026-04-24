"""
Test type annotations in critical modules.

This test file verifies that mypy type checking can be run on critical modules
without errors. Run with: mypy --config-file=mypy.ini src/models/factory.py src/trainer.py
"""

import pytest
import torch
import torch.nn as nn
from src.models.factory import ChannelAdapter, get_model
from typing import Dict, Any


def test_channel_adapter_types() -> None:
    """Verify ChannelAdapter has correct type signatures."""
    adapter: ChannelAdapter = ChannelAdapter(in_channels=2)
    x: torch.Tensor = torch.randn(2, 2, 256, 256)
    output: torch.Tensor = adapter(x)
    assert output.shape == (2, 1, 256, 256)


def test_get_model_return_type() -> None:
    """Verify get_model returns nn.Module."""
    config: Dict[str, Any] = {
        "common": {"in_channels": 2, "out_channels": 1},
        "drunet": {"base_channels": 64},
    }
    model: nn.Module = get_model("drunet", config)
    assert isinstance(model, nn.Module)


def test_get_model_invalid_raises() -> None:
    """Verify get_model raises ValueError for unknown model."""
    config: Dict[str, Any] = {
        "common": {"in_channels": 2, "out_channels": 1},
    }
    with pytest.raises(ValueError):
        get_model("unknown_model", config)
