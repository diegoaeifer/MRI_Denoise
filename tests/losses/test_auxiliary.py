import pytest
import torch
import torch.nn as nn
from src.losses.auxiliary import MCSURELoss

class DummyModel(nn.Module):
    def __init__(self, output_val=1.0):
        super().__init__()
        self.output_val = output_val

    def forward(self, x):
        # We need a model that returns something dependent on x to have non-zero divergence
        # h(y) = y_img * 0.5 + output_val
        y_img = x[:, 0:1, :, :]
        return y_img * 0.5 + self.output_val

def test_mcsure_loss_with_sigma_value():
    loss_fn = MCSURELoss(sigma=0.1, eps=1e-4)
    model = DummyModel(output_val=0.0)

    noisy_input = torch.ones((1, 2, 8, 8)) # B, C(image+sigma), H, W
    predicted_output = model(noisy_input) # Will be 0.5

    loss = loss_fn(model, noisy_input, predicted_output)

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    # Values sanity check
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_mcsure_loss_with_sigma_map():
    loss_fn = MCSURELoss(eps=1e-4)
    model = DummyModel(output_val=0.0)

    noisy_input = torch.ones((1, 2, 8, 8))
    sigma_map = torch.full((1, 1, 8, 8), 0.1)

    predicted_output = model(noisy_input)

    loss = loss_fn(model, noisy_input, predicted_output, sigma_map=sigma_map)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_mcsure_loss_missing_sigma():
    loss_fn = MCSURELoss(eps=1e-4)
    model = DummyModel()

    noisy_input = torch.ones((1, 2, 8, 8))
    predicted_output = model(noisy_input)

    with pytest.raises(ValueError, match="SURE requires a sigma value or map."):
        loss_fn(model, noisy_input, predicted_output)

from src.losses.auxiliary import CharbonnierLoss, VGGPerceptualLoss

def test_charbonnier_loss():
    loss_fn = CharbonnierLoss(eps=1e-3)
    x = torch.ones((1, 1, 8, 8))
    y = torch.ones((1, 1, 8, 8))
    loss = loss_fn(x, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    # Should be exactly eps if x == y
    assert torch.isclose(loss, torch.tensor(1e-3))

    y = torch.zeros((1, 1, 8, 8))
    loss_diff = loss_fn(x, y)
    assert loss_diff > loss

def test_vgg_perceptual_loss():
    loss_fn = VGGPerceptualLoss(layer_name='relu3_3')
    x = torch.ones((1, 1, 32, 32))
    y = torch.ones((1, 1, 32, 32))
    loss = loss_fn(x, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    y = torch.zeros((1, 1, 32, 32))
    loss_diff = loss_fn(x, y)
    assert loss_diff > 0

def test_vgg_perceptual_loss_other_layer():
    loss_fn = VGGPerceptualLoss(layer_name='other')
    x = torch.ones((1, 1, 32, 32))
    y = torch.ones((1, 1, 32, 32))
    loss = loss_fn(x, y)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)
