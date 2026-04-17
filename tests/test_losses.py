import torch
import pytest
import torch.nn.functional as F
from src.losses.composite import EPILoss

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
