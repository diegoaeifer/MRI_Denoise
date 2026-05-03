"""
Tests for the MONAI-native CompositeLoss.

Tests spatial_dims=2 and spatial_dims=3 paths, weight gating, gradient flow,
and no-NaN stability.
"""

import pytest
import torch


@pytest.fixture
def pair_2d():
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    return pred, target


@pytest.fixture
def pair_3d():
    pred = torch.rand(2, 1, 32, 32, 8)
    target = torch.rand(2, 1, 32, 32, 8)
    return pred, target


class TestCompositeLossMonai:
    def _build(self, spatial_dims=2, weights=None):
        from src.mri_denoise.losses.composite import CompositeLoss
        weights = weights or {"l1": 1.0}
        return CompositeLoss(spatial_dims=spatial_dims, weights=weights)

    def test_import(self):
        from src.mri_denoise.losses.composite import CompositeLoss
        assert CompositeLoss is not None

    def test_l1_only_returns_scalar(self, pair_2d):
        pred, target = pair_2d
        fn = self._build(weights={"l1": 1.0})
        loss, details = fn(pred, target)
        assert loss.ndim == 0
        assert isinstance(details, dict)

    def test_details_dict_contains_l1(self, pair_2d):
        pred, target = pair_2d
        fn = self._build(weights={"l1": 1.0})
        _, details = fn(pred, target)
        assert "l1" in details

    def test_no_nan_random_input_2d(self, pair_2d):
        pred, target = pair_2d
        fn = self._build(weights={"l1": 1.0, "ssim": 0.1, "psnr": 0.05})
        loss, _ = fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_no_nan_random_input_3d(self, pair_3d):
        pred, target = pair_3d
        fn = self._build(spatial_dims=3, weights={"l1": 1.0})
        loss, _ = fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows(self, pair_2d):
        pred, target = pair_2d
        pred = pred.requires_grad_(True)
        fn = self._build(weights={"l1": 1.0})
        loss, _ = fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

    def test_zero_weight_component_skipped(self, pair_2d):
        pred, target = pair_2d
        # Both configs have l1=1.0; the ssim=0.0 should be a no-op.
        fn_a = self._build(weights={"l1": 1.0, "ssim": 0.0})
        fn_b = self._build(weights={"l1": 1.0})
        loss_a, _ = fn_a(pred, target)
        loss_b, _ = fn_b(pred, target)
        assert torch.allclose(loss_a, loss_b)

    def test_loss_lower_when_pred_equals_target(self):
        from src.mri_denoise.losses.composite import CompositeLoss
        target = torch.ones(1, 1, 64, 64) * 0.5
        fn = CompositeLoss(spatial_dims=2, weights={"l1": 1.0})
        loss_bad, _ = fn(torch.zeros(1, 1, 64, 64), target)
        loss_good, _ = fn(target.clone(), target)
        assert loss_good < loss_bad

    def test_charbonnier_weight(self, pair_2d):
        pred, target = pair_2d
        fn = self._build(weights={"charbonnier": 1.0})
        loss, details = fn(pred, target)
        assert "charbonnier" in details
        assert not torch.isnan(loss)

    def test_psnr_weight(self, pair_2d):
        pred, target = pair_2d
        fn = self._build(weights={"psnr": 0.1})
        loss, details = fn(pred, target)
        assert "psnr" in details
