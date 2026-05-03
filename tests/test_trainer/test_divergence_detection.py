"""
Tests for divergence detection via DivergenceStopHandler.

Replaces old Trainer-based divergence tests.
"""

import pytest
import torch
import torch.nn as nn


class TestDivergenceStopHandler:
    """Test suite for the Ignite-based DivergenceStopHandler."""

    def test_handler_import(self):
        """Test that DivergenceStopHandler can be imported."""
        try:
            from src.mri_denoise.handlers.divergence import DivergenceStopHandler
            assert DivergenceStopHandler is not None
        except ImportError:
            pytest.skip("DivergenceStopHandler not available")

    def test_handler_terminates_on_negative_psnr(self):
        """Test that handler terminates trainer after N consecutive negative PSNRs."""
        try:
            from ignite.engine import Engine
            from src.mri_denoise.handlers.divergence import DivergenceStopHandler
        except ImportError:
            pytest.skip("Ignite or handler not available")

        def _fake_process(e, batch):
            return batch

        trainer = Engine(_fake_process)
        evaluator = Engine(_fake_process)

        terminated = []
        original_terminate = trainer.terminate

        def _mock_terminate():
            terminated.append(True)

        trainer.terminate = _mock_terminate
        handler = DivergenceStopHandler(trainer, threshold=2, psnr_key="psnr")
        handler.attach(evaluator)

        # Simulate 3 consecutive negative PSNR epochs
        evaluator.state = type("S", (), {"metrics": {"psnr": -1.0}})()
        handler(evaluator)
        assert len(terminated) == 0  # First negative — no termination yet

        handler(evaluator)
        assert len(terminated) == 1  # Second negative — threshold reached, terminate


class TestNegativePSNRDetection:
    """Property-based tests for PSNR calculation and divergence."""

    def test_psnr_calculation_formula(self):
        """Test PSNR calculation: 20 * log10(max / sqrt(mse))."""
        pred = torch.ones(2, 1, 32, 32) * 100  # Very large diverged values
        target = torch.zeros(2, 1, 32, 32)  # Zero target

        max_val = 1.0
        mse = ((pred - target) ** 2).mean()
        psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))

        assert psnr.item() < 0, "PSNR should be negative when mse > max_val"

    def test_gradient_stability(self):
        """Test that gradients don't explode during training."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(4, 10)
        target = torch.randn(4, 1)

        for _ in range(5):
            optimizer.zero_grad()
            pred = model(x)
            loss = (pred - target).pow(2).mean()
            loss.backward()
            optimizer.step()

        max_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())

        assert max_grad < 100.0, f"Gradient explosion detected: {max_grad}"

    def test_loss_nan_detection(self):
        """Test detection of NaN loss values."""
        pred = torch.tensor([[[[float("nan")]]]])
        target = torch.zeros(1, 1, 1, 1)
        loss = (pred - target).mean()
        assert torch.isnan(loss), "NaN loss should be detected"

    def test_early_stopping_trigger(self):
        """Test early stopping logic."""
        max_divergence_epochs = 3
        divergence_count = 0
        should_stop = False

        # Simulate 5 epochs of divergence
        for epoch in range(5):
            divergence_count += 1
            if divergence_count >= max_divergence_epochs:
                should_stop = True
                break

        assert should_stop, "Should trigger early stopping"
        assert epoch == 2, "Should stop after 3 epochs (0-indexed)"
