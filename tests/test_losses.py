import pytest
import torch
from src.losses.auxiliary import CharbonnierLoss
import torch.nn.functional as F
from src.losses.composite import EPILoss

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

def test_epi_loss_identical():
    loss_fn = EPILoss()
    # Create a random target tensor with shape (B, C, H, W)
    # Using deterministic seed for reproducibility
    torch.manual_seed(42)
    target = torch.rand(2, 3, 64, 64)
    pred = target.clone()

    loss = loss_fn(pred, target)

    # Loss is 1.0 - EPI. For identical tensors, EPI should be 1.0, so loss should be 0.0
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5), f"Expected loss 0.0 for identical inputs, got {loss.item()}"

def test_epi_loss_blurred():
    loss_fn = EPILoss()
    torch.manual_seed(42)
    target = torch.rand(2, 3, 64, 64)

    # Create a blurred prediction using average pooling
    # This degrades high-frequency edge information
    pred = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    loss_identical = loss_fn(target, target)
    loss_blurred = loss_fn(pred, target)

    # Blurred image should have lower EPI, meaning higher loss
    assert loss_blurred > loss_identical, f"Expected higher loss for blurred image ({loss_blurred.item()}) compared to identical image ({loss_identical.item()})"
