"""
Tests for SwinUNETRDenoising:
- (B, 2, H, W) → (B, 1, H, W) forward pass
- Encoder freeze/unfreeze via set_epoch
- get_layerwise_lr_groups produces valid param groups
"""

import pytest
import torch


@pytest.fixture(scope="module")
def model_2d():
    pytest.importorskip("monai", reason="MONAI required for SwinUNETRDenoising")
    from src.mri_denoise.networks.swinunetr_denoising import SwinUNETRDenoising
    return SwinUNETRDenoising(
        spatial_dims=2,
        img_size=(64, 64),
        feature_size=12,  # smallest valid feature_size for fast test
        freeze_encoder_epochs=2,
    )


class TestSwinUNETRDenoising:
    def test_forward_shape(self, model_2d):
        model_2d.eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model_2d(x)
        assert y.shape == (1, 1, 64, 64), f"Expected (1,1,64,64), got {y.shape}"

    def test_encoder_frozen_at_epoch_0(self, model_2d):
        model_2d.set_epoch(0)
        frozen = all(
            not p.requires_grad
            for p in model_2d.backbone.swinViT.parameters()
        )
        assert frozen, "Encoder should be frozen at epoch 0"

    def test_encoder_unfrozen_after_threshold(self, model_2d):
        model_2d.set_epoch(2)  # freeze_encoder_epochs=2
        unfrozen = any(
            p.requires_grad
            for p in model_2d.backbone.swinViT.parameters()
        )
        assert unfrozen, "Encoder should be unfrozen after freeze_encoder_epochs"

    def test_set_epoch_idempotent(self, model_2d):
        model_2d.set_epoch(5)
        model_2d.set_epoch(5)  # calling twice should not error

    def test_output_is_finite(self, model_2d):
        model_2d.eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model_2d(x)
        assert torch.isfinite(y).all()


class TestLayerwiseLR:
    def test_groups_have_positive_lr(self):
        from src.mri_denoise.handlers.layerwise_lr import get_layerwise_lr_groups
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        groups = get_layerwise_lr_groups(model, base_lr=1e-4, decay_factor=0.9)
        assert len(groups) > 0
        for g in groups:
            assert g["lr"] > 0
            assert len(g["params"]) == 1

    def test_lr_decays_monotonically(self):
        from src.mri_denoise.handlers.layerwise_lr import get_layerwise_lr_groups
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 4)
        )
        groups = get_layerwise_lr_groups(model, base_lr=1e-3, decay_factor=0.8)
        lrs = [g["lr"] for g in groups]
        # LRs should be non-increasing (first group = deepest = highest LR)
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))
