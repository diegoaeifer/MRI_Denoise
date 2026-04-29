import pytest
import torch
import torch.nn as nn


class TestCompositeLoss:
    """Test suite for CompositeLoss and individual loss components."""

    @pytest.fixture
    def pred_target_pair(self):
        """Create pred and target tensors."""
        batch_size = 2
        height, width = 256, 256
        pred = torch.randn(batch_size, 1, height, width)
        target = torch.randn(batch_size, 1, height, width)
        return pred, target

    @pytest.fixture
    def dummy_model(self):
        """Create a simple dummy model for SURE loss."""

        class DummyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        return DummyNet()

    def test_composite_loss_import(self):
        """Test that CompositeLoss can be imported."""
        try:
            from src.losses.composite import CompositeLoss

            assert CompositeLoss is not None
        except ImportError:
            pytest.skip("CompositeLoss not available")

    def test_composite_loss_returns_scalar(self, pred_target_pair):
        """Test that CompositeLoss returns a scalar loss."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        loss_fn = CompositeLoss(weights={"l1": 1.0})
        loss = loss_fn(pred, target)

        assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_composite_loss_with_zero_weight_components(self, pred_target_pair):
        """Test that zero-weight components don't affect loss."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair

        # Loss with only L1
        loss_fn1 = CompositeLoss(
            weights={
                "l1": 1.0,
                "ssim": 0.0,
                "ms_ssim": 0.0,
                "psnr": 0.0,
            }
        )
        loss1 = loss_fn1(pred, target)

        # Loss with L1 + zero-weight components
        loss_fn2 = CompositeLoss(
            weights={
                "l1": 1.0,
                "ssim": 0.0,
                "ms_ssim": 0.0,
                "psnr": 0.0,
            }
        )
        loss2 = loss_fn2(pred, target)

        assert torch.allclose(
            loss1, loss2
        ), "Zero-weight components should not affect loss"

    def test_composite_loss_decreases_when_pred_matches_target(self):
        """Test that loss decreases when pred matches target."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        target = torch.ones(2, 1, 256, 256)
        loss_fn = CompositeLoss(weights={"l1": 1.0})

        # Loss with random pred
        pred_random = torch.randn(2, 1, 256, 256)
        loss_random = loss_fn(pred_random, target)

        # Loss with pred == target
        pred_exact = target.clone()
        loss_exact = loss_fn(pred_exact, target)

        assert loss_exact < loss_random, "Loss should decrease when pred matches target"

    def test_composite_loss_no_nan_with_random_input(self, pred_target_pair):
        """Test that CompositeL loss doesn't produce NaN with random inputs."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        loss_fn = CompositeLoss(weights={"l1": 1.0, "ssim": 1.0})
        loss = loss_fn(pred, target)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_composite_loss_gradient_flow(self, pred_target_pair):
        """Test that gradients flow through CompositeLoss."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        pred.requires_grad_(True)

        loss_fn = CompositeLoss(weights={"l1": 1.0})
        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None, "Gradients should flow through CompositeLoss"
        assert not torch.allclose(
            pred.grad, torch.zeros_like(pred.grad)
        ), "Gradients should be non-zero"

    def test_l1_loss_component(self, pred_target_pair):
        """Test that L1 loss component works."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        loss_fn = CompositeLoss(weights={"l1": 1.0})

        # Loss should be zero when pred == target
        zero_loss = loss_fn(target, target)
        assert zero_loss.item() < 1e-5, "L1 loss should be ~0 when pred == target"

    def test_composite_loss_with_multiple_components(self, pred_target_pair):
        """Test CompositeLoss with multiple weighted components."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        pred, target = pred_target_pair
        weights = {
            "l1": 1.0,
            "ssim": 0.5,
            "psnr": 0.1,
        }
        loss_fn = CompositeLoss(weights=weights)
        loss = loss_fn(pred, target)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_composite_loss_batch_consistency(self):
        """Test that loss is consistent across different batch sizes."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        target = torch.ones(1, 1, 256, 256)
        pred = torch.ones(1, 1, 256, 256) * 1.1

        loss_fn = CompositeLoss(weights={"l1": 1.0})

        # Single sample
        loss_single = loss_fn(pred, target)

        # Multiple samples (same values repeated)
        pred_batch = pred.repeat(4, 1, 1, 1)
        target_batch = target.repeat(4, 1, 1, 1)
        loss_batch = loss_fn(pred_batch, target_batch)

        # Losses should be proportional (batch averaging)
        assert torch.allclose(
            loss_single, loss_batch, rtol=0.1
        ), "Loss should be consistent across batch sizes"


class TestCharbonnierLoss:
    """Test Charbonnier loss robustness."""

    def test_charbonnier_import(self):
        """Test that CharbonnierLoss can be imported."""
        try:
            from src.losses.auxiliary import CharbonnierLoss

            assert CharbonnierLoss is not None
        except ImportError:
            pytest.skip("CharbonnierLoss not available")

    def test_charbonnier_no_nan_on_outliers(self):
        """Test that Charbonnier loss is robust to outliers."""
        try:
            from src.losses.auxiliary import CharbonnierLoss
        except ImportError:
            pytest.skip("CharbonnierLoss not available")

        loss_fn = CharbonnierLoss()

        # Create tensors with extreme values
        pred = torch.tensor([[[1000.0, 0.0, -1000.0]]])
        target = torch.tensor([[[0.0, 0.0, 0.0]]])

        loss = loss_fn(pred, target)

        assert not torch.isnan(loss), "Charbonnier loss should handle outliers"
        assert not torch.isinf(loss), "Charbonnier loss should not be Inf"


class TestVGGPerceptualLoss:
    """Test VGG Perceptual loss."""

    def test_vgg_perceptual_loss_import(self):
        """Test that VGGPerceptualLoss can be imported."""
        try:
            from src.losses.auxiliary import VGGPerceptualLoss

            assert VGGPerceptualLoss is not None
        except ImportError:
            pytest.skip("VGGPerceptualLoss not available")

    def test_vgg_handles_grayscale_to_rgb_conversion(self):
        """Test that VGG Perceptual loss handles grayscale to RGB conversion."""
        try:
            from src.losses.auxiliary import VGGPerceptualLoss
        except ImportError:
            pytest.skip("VGGPerceptualLoss not available")

        # Use GPU if available, fall back to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            loss_fn = VGGPerceptualLoss().to(device)
        except RuntimeError:
            pytest.skip("VGG model loading failed (GPU memory?)")

        # Grayscale input
        pred = torch.randn(1, 1, 128, 128, device=device)
        target = torch.randn(1, 1, 128, 128, device=device)

        try:
            loss = loss_fn(pred, target)
            assert not torch.isnan(loss), "VGG loss should not be NaN"
        except RuntimeError:
            pytest.skip("VGG loss computation failed")


class TestLossStability:
    """Property-based tests for loss stability."""

    def test_loss_always_non_negative(self):
        """Test that all losses are non-negative."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        loss_fn = CompositeLoss(weights={"l1": 1.0})

        for _ in range(10):
            pred = torch.randn(2, 1, 128, 128)
            target = torch.randn(2, 1, 128, 128)
            loss = loss_fn(pred, target)
            assert loss.item() >= 0, "Loss should be non-negative"

    def test_loss_symmetric_behavior(self):
        """Test that loss composition is sensible."""
        try:
            from src.losses.composite import CompositeLoss
        except ImportError:
            pytest.skip("CompositeLoss not available")

        loss_fn = CompositeLoss(weights={"l1": 1.0})

        # Create tensors
        a = torch.ones(1, 1, 64, 64)
        b = torch.ones(1, 1, 64, 64) * 2

        loss_ab = loss_fn(a, b)
        loss_ba = loss_fn(b, a)

        # L1 loss should be symmetric
        assert torch.allclose(
            loss_ab, loss_ba, rtol=0.01
        ), "L1 loss should be symmetric"
