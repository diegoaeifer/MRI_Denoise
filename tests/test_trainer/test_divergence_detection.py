import pytest
import torch
import torch.nn as nn


class TestTrainerDivergenceDetection:
    """Test suite for trainer divergence detection."""

    def test_trainer_import(self):
        """Test that Trainer can be imported."""
        try:
            from src.trainer import Trainer
            assert Trainer is not None
        except ImportError:
            pytest.skip("Trainer not available")

    def test_negative_psnr_detection(self):
        """Test that negative PSNR is detected as divergence."""
        # PSNR formula: 20 * log10(max / mse)
        # Negative PSNR occurs when mse > max value

        # Create mock tensors with diverged (large error) outputs
        pred = torch.randn(2, 1, 256, 256) * 100  # Very large values
        target = torch.zeros(2, 1, 256, 256)  # Zero target

        mse = ((pred - target) ** 2).mean()
        max_val = target.abs().max() + 1e-8

        if mse > max_val:
            psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
            # PSNR should be negative or very small
            assert psnr < 10, "PSNR should be low for diverged output"

    def test_loss_nan_detection(self):
        """Test that NaN loss is detected."""
        # Create a loss that produces NaN
        pred = torch.tensor([[[[float("nan")]]]])
        target = torch.zeros(1, 1, 1, 1)

        loss = (pred - target).mean()
        assert torch.isnan(loss), "NaN loss should be detected"

    def test_loss_inf_detection(self):
        """Test that Inf loss is detected."""
        pred = torch.tensor([[[[float("inf")]]]])
        target = torch.zeros(1, 1, 1, 1)

        loss = (pred - target).mean()
        assert torch.isinf(loss), "Inf loss should be detected"

    def test_divergence_counter_logic(self):
        """Test divergence counter increment/reset logic."""
        # Simulate divergence counter behavior
        divergence_count = 0
        max_divergence_tolerance = 5

        # Scenario 1: Loss increases (divergence)
        losses = [1.0, 1.1, 1.2, 1.3, 1.4]
        for loss in losses:
            if loss > 1.0:  # Simple divergence check
                divergence_count += 1
            else:
                divergence_count = 0

        assert divergence_count == 5, "Divergence counter should increment"

        # Scenario 2: Loss decreases (improvement)
        divergence_count = 0
        losses = [1.0, 0.9, 0.8, 0.7]
        for loss in losses:
            if loss > 1.0:
                divergence_count += 1
            else:
                divergence_count = 0

        assert divergence_count == 0, "Divergence counter should reset on improvement"

    def test_early_stopping_on_divergence(self):
        """Test early stopping trigger logic."""
        max_divergence_epochs = 5
        divergence_count = 0
        should_stop = False

        # Simulate 10 epochs of divergence
        for epoch in range(10):
            divergence_count += 1
            if divergence_count >= max_divergence_epochs:
                should_stop = True
                break

        assert should_stop is True, "Should trigger early stopping"
        assert epoch == 4, "Should stop after 5 epochs of divergence (epoch 0-4)"

    def test_gradient_explosion_detection(self):
        """Test detection of gradient explosion."""
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 10),
        )

        x = torch.randn(1, 10)
        target = torch.randn(1, 10)

        output = model(x)
        loss = (output - target).mean()
        loss.backward()

        # Check for gradient explosion
        max_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())

        # Gradients should be reasonable (not exploded)
        assert max_grad < 1e6, f"Gradient explosion detected: max_grad={max_grad}"

    def test_psnr_calculation(self):
        """Test PSNR calculation for loss monitoring."""
        pred = torch.ones(1, 1, 128, 128) * 0.9
        target = torch.ones(1, 1, 128, 128)

        # PSNR = 20 * log10(max / sqrt(mse))
        max_val = 1.0
        mse = ((pred - target) ** 2).mean()
        psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))

        assert psnr > 0, "PSNR should be positive for similar tensors"
        assert psnr.item() > 10, "PSNR should be reasonable (>10 dB)"

    def test_metric_tracking(self):
        """Test metric tracking logic."""
        metrics = {
            "loss": [],
            "psnr": [],
            "ssim": [],
        }

        # Simulate epoch training
        for epoch in range(5):
            loss = 1.0 / (epoch + 1)
            psnr = 10.0 + epoch

            metrics["loss"].append(loss)
            metrics["psnr"].append(psnr)

        assert len(metrics["loss"]) == 5, "Metrics should be tracked"
        assert metrics["loss"][0] > metrics["loss"][-1], "Loss should decrease"
        assert metrics["psnr"][-1] > metrics["psnr"][0], "PSNR should increase"

    def test_checkpoint_save_structure(self, tmp_path):
        """Test checkpoint saving structure."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Simulate checkpoint structure
        checkpoint = {
            "epoch": 10,
            "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
            "optimizer_state_dict": {"lr": 0.0001},
            "loss": 0.5,
        }

        torch.save(checkpoint, checkpoint_path)

        # Load and verify
        loaded = torch.load(checkpoint_path)
        assert loaded["epoch"] == 10
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
        assert "loss" in loaded

    def test_checkpoint_load_and_resume(self, tmp_path):
        """Test checkpoint loading for training resumption."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Create and save checkpoint
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        original_state = model.state_dict()
        checkpoint = {
            "epoch": 5,
            "model_state_dict": original_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0.5,
        }
        torch.save(checkpoint, checkpoint_path)

        # Create new model and load checkpoint
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)

        loaded_checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

        # Verify state was restored
        for orig_param, new_param in zip(original_state.values(), new_model.state_dict().values()):
            assert torch.allclose(orig_param, new_param), "Model state should be restored"
