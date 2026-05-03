"""
Tests for MONAI-based CompositeLoss.

Tests the new CompositeLoss that uses MONAI's SSIMLoss, PerceptualLoss, etc.
"""

import pytest
import torch
import torch.nn as nn


class TestCompositeLossMonai:
    """Test suite for MONAI-based CompositeLoss."""

    @pytest.fixture
    def pred_target_pair(self):
        """Create pred and target tensors."""
        batch_size = 2
        height, width = 256, 256
        pred = torch.randn(batch_size, 1, height, width)
        target = torch.randn(batch_size, 1, height, width)
        return pred, target

    def test_composite_loss_import(self):
        """Test that CompositeLoss can be imported."""
        try:
            from src.mri_denoise.losses.composite import CompositeLoss
            assert CompositeLoss is not None
        except ImportError:
            pytest.skip("CompositeLoss not available")

    def test_composite_loss_returns_loss_and_details(self, pred_target_pair):
        """Test that CompositeLoss returns (loss, details_dict)."""
        try:
            from src.mri_denoise.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        loss_fn = CompositeLoss(spatial_dims=2, weights={"l1": 1.0})
        loss, details = loss_fn(pred, target)

        assert loss.ndim == 0, f"Expected scalar loss, got {loss.shape}"
        assert isinstance(details, dict), "Second return should be dict"
        assert "l1" in details

    def test_composite_loss_no_nan_random_input(self, pred_target_pair):
        """Test that CompositeLoss doesn't produce NaN with random inputs."""
        try:
            from src.mri_denoise.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        loss_fn = CompositeLoss(spatial_dims=2, weights={"l1": 1.0, "ssim": 0.1})
        loss, _ = loss_fn(pred, target)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_composite_loss_gradient_flow(self, pred_target_pair):
        """Test that gradients flow through CompositeLoss."""
        try:
            from src.mri_denoise.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        pred = pred.requires_grad_(True)
        loss_fn = CompositeLoss(spatial_dims=2, weights={"l1": 1.0})
        loss, _ = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None, "Gradients should flow through loss"
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

    def test_composite_loss_lower_for_perfect_prediction(self):
        """Test that loss is lower when pred matches target."""
        try:
            from src.mri_denoise.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        target = torch.ones(1, 1, 64, 64) * 0.5
        loss_fn = CompositeLoss(spatial_dims=2, weights={"l1": 1.0})

        loss_bad, _ = loss_fn(torch.zeros(1, 1, 64, 64), target)
        loss_good, _ = loss_fn(target.clone(), target)

        assert loss_good < loss_bad, "Loss should be lower for perfect prediction"

    def test_composite_loss_spatial_dims_3(self):
        """Test CompositeLoss with spatial_dims=3."""
        try:
            from src.mri_denoise.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred = torch.randn(1, 1, 32, 32, 16)
        target = torch.randn(1, 1, 32, 32, 16)
        loss_fn = CompositeLoss(spatial_dims=3, weights={"l1": 1.0})
        loss, _ = loss_fn(pred, target)

        assert not torch.isnan(loss), "3D loss should not be NaN"
        assert loss.ndim == 0, "Should return scalar"

    def test_zero_weight_component_skipped(self):
        """Test that zero-weight components don't affect loss."""
        try:
            from src.mri_denoise.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)

        fn_a = CompositeLoss(spatial_dims=2, weights={"l1": 1.0, "ssim": 0.0})
        fn_b = CompositeLoss(spatial_dims=2, weights={"l1": 1.0})

        loss_a, _ = fn_a(pred, target)
        loss_b, _ = fn_b(pred, target)

        assert torch.allclose(loss_a, loss_b, rtol=0.05), "Zero-weight components should not affect loss"
