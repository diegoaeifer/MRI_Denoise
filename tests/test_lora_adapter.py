import torch
import pytest
from models.lora_adapter import expand_input_channels, attach_lora, LoRAConv2d, LoRAWrapper


def _simple_model():
    """Small conv net with known input channels."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(2, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 1, 1),
    )


def test_expand_input_channels_shape():
    """After expansion, first conv accepts new_in_channels."""
    model = _simple_model()
    model = expand_input_channels(model, new_in_channels=3)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 1, 32, 32)


def test_expand_input_channels_identity_at_init():
    """
    With new channels zeroed, model output for ch0+ch1 should match
    original model output (forward pass is identical for existing channels).
    """
    torch.manual_seed(0)
    model = _simple_model()
    x2 = torch.randn(1, 2, 32, 32)

    with torch.no_grad():
        expected = model(x2)

    model_expanded = expand_input_channels(model, new_in_channels=3)
    x3 = torch.cat([x2, torch.zeros(1, 1, 32, 32)], dim=1)

    with torch.no_grad():
        actual = model_expanded(x3)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Expected output unchanged after expansion, "
        f"got max diff {(expected - actual).abs().max()}"
    )


def test_attach_lora_params_added():
    """LoRA adds parameters (lora_A, lora_B) to conv layers."""
    model = _simple_model()
    lora_model = attach_lora(model, rank=4)
    lora_param_names = [n for n, _ in lora_model.named_parameters() if "lora_" in n]
    assert len(lora_param_names) > 0


def test_lora_output_same_at_init():
    """LoRA B initialized to zeros => output identical to base at init."""
    torch.manual_seed(42)
    model = _simple_model()
    x = torch.randn(1, 2, 32, 32)
    with torch.no_grad():
        base_out = model(x)

    lora_model = attach_lora(model, rank=4)
    with torch.no_grad():
        lora_out = lora_model(x)

    assert torch.allclose(base_out, lora_out, atol=1e-5)


def test_lora_wrapper_freezes_base():
    """LoRAWrapper: only lora_ params should be trainable."""
    model = _simple_model()
    wrapper = LoRAWrapper(model, new_in_channels=3, rank=4)

    trainable = [n for n, p in wrapper.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable), (
        f"Non-LoRA trainable params found: "
        f"{[n for n in trainable if 'lora_' not in n]}"
    )


def test_lora_wrapper_forward_shape():
    model = _simple_model()
    wrapper = LoRAWrapper(model, new_in_channels=3, rank=4)
    x = torch.randn(1, 3, 32, 32)
    out = wrapper(x)
    assert out.shape == (1, 1, 32, 32)


def test_lora_wrapper_trainable_params_fewer_than_total():
    model = _simple_model()
    wrapper = LoRAWrapper(model, new_in_channels=3, rank=4)
    assert wrapper.trainable_params() < wrapper.total_params()
    assert wrapper.trainable_params() > 0


def test_lora_conv2d_gradients_flow():
    """Gradients should flow through LoRA params."""
    conv = torch.nn.Conv2d(2, 8, 3, padding=1)
    lora_conv = LoRAConv2d(conv, rank=4)
    x = torch.randn(1, 2, 16, 16, requires_grad=True)
    out = lora_conv(x)
    loss = out.sum()
    loss.backward()
    assert lora_conv.lora_A.grad is not None
    assert lora_conv.lora_B.grad is not None
