import pytest
import torch
from src.losses.auxiliary import CharbonnierLoss

def test_charbonnier_loss_identical_inputs():
    loss_fn = CharbonnierLoss(eps=1e-3)
    x = torch.ones(2, 3, 4, 4)
    y = torch.ones(2, 3, 4, 4)

    loss = loss_fn(x, y)

    # When x == y, diff is 0, so loss = sqrt(eps^2) = eps
    assert torch.isclose(loss, torch.tensor(1e-3)), f"Expected 1e-3, got {loss.item()}"

def test_charbonnier_loss_mathematical_correctness():
    loss_fn = CharbonnierLoss(eps=1e-3)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([0.1, 0.2, -0.1])

    loss = loss_fn(x, y)

    diff = x - y
    expected_loss = torch.mean(torch.sqrt(diff * diff + 1e-6))

    assert torch.isclose(loss, expected_loss)

def test_charbonnier_loss_eps_zero():
    loss_fn = CharbonnierLoss(eps=0.0)
    x = torch.tensor([1.0, 2.0, -3.0])
    y = torch.tensor([0.0, 2.0, -1.0])

    loss = loss_fn(x, y)
    assert torch.isclose(loss, torch.tensor(1.0))

def test_charbonnier_loss_varying_shapes():
    loss_fn = CharbonnierLoss(eps=1e-3)

    # 2D tensor
    x_2d = torch.randn(4, 4)
    y_2d = torch.randn(4, 4)
    loss_2d = loss_fn(x_2d, y_2d)
    assert loss_2d.ndim == 0 # mean returns a scalar

    # 4D tensor
    x_4d = torch.randn(2, 3, 16, 16)
    y_4d = torch.randn(2, 3, 16, 16)
    loss_4d = loss_fn(x_4d, y_4d)
    assert loss_4d.ndim == 0

    # Broadcastable tensors
    x_b = torch.randn(2, 3, 4, 4)
    y_b = torch.randn(1, 3, 1, 1)
    loss_b = loss_fn(x_b, y_b)
    assert loss_b.ndim == 0
