import torch
import pytest
from src.models.foura_adapter import FouRAConv2d, attach_foura, FouRAWrapper

def test_foura_conv2d_forward():
    conv = torch.nn.Conv2d(32, 64, 3, padding=1)
    foura = FouRAConv2d(conv, rank=8, alpha=16.0)
    x = torch.randn(2, 32, 64, 64)
    out = foura(x)
    assert out.shape == (2, 64, 64, 64)

def test_foura_only_adapters_trainable():
    conv = torch.nn.Conv2d(32, 64, 3, padding=1)
    foura = FouRAConv2d(conv, rank=8)
    trainable = [n for n, p in foura.named_parameters() if p.requires_grad]
    assert all("foura_" in n for n in trainable), f"Non-adapter params trainable: {trainable}"

def test_foura_wrapper_parameter_count():
    from src.models.nafnet import NAFNet
    model = NAFNet(img_channel=2, width=32)
    wrapped = FouRAWrapper(model, rank=16)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    print(f"Total: {total:,}  Trainable (FouRA): {trainable:,}  Ratio: {trainable/total:.2%}")
    assert trainable < total * 0.10, "FouRA should train <10% of parameters"

def test_foura_matches_baseline_at_init():
    """At initialization B=0 so FouRA output equals original Conv2d output."""
    torch.manual_seed(42)
    conv = torch.nn.Conv2d(16, 32, 3, padding=1)
    foura = FouRAConv2d(conv, rank=4)
    x = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        base_out = conv(x)
        foura_out = foura(x)
    assert torch.allclose(base_out, foura_out, atol=1e-5), "FouRA should match baseline at init (B=0)"
