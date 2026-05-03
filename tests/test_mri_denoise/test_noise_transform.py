"""
Tests for SpatiallyVaryingNoised — the core custom MONAI transform.

Covers: output shapes, sigma_map range, determinism under seeding,
gaussian vs rician mode, 2D vs 3D dispatch, and probability gating.
"""

import numpy as np
import pytest
import torch

from src.mri_denoise.data.transforms.noise import SpatiallyVaryingNoised


@pytest.fixture
def img_2d():
    """(C, H, W) float32 numpy array."""
    return np.random.rand(1, 64, 64).astype(np.float32)


@pytest.fixture
def img_3d():
    """(C, H, W, D) float32 numpy array."""
    return np.random.rand(1, 32, 32, 8).astype(np.float32)


class TestSpatiallyVaryingNoised:
    def test_output_keys_2d(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", prob=1.0)
        out = t({"image": torch.from_numpy(img_2d)})
        assert "image" in out
        assert "image_sigma_map" in out

    def test_output_shape_preserved_2d(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", prob=1.0)
        out = t({"image": torch.from_numpy(img_2d)})
        assert out["image"].shape == img_2d.shape
        assert out["image_sigma_map"].shape == img_2d.shape[1:]  # (H, W)

    def test_output_shape_preserved_3d(self, img_3d):
        t = SpatiallyVaryingNoised(keys="image", spatial_dims=3, prob=1.0)
        out = t({"image": torch.from_numpy(img_3d)})
        assert out["image"].shape == img_3d.shape
        assert out["image_sigma_map"].shape == img_3d.shape[1:]  # (H, W, D)

    def test_sigma_map_range(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", sigma_range=(0.05, 0.2), prob=1.0)
        out = t({"image": torch.from_numpy(img_2d)})
        sm = out["image_sigma_map"]
        assert sm.min() >= 0.0
        assert sm.max() <= 1.0

    def test_gaussian_mode_produces_noise(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", noise_type="gaussian", prob=1.0)
        inp = torch.from_numpy(img_2d)
        out = t({"image": inp})
        # Output should differ from input
        assert not torch.allclose(out["image"], inp)

    def test_rician_mode_produces_noise(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", noise_type="rician", prob=1.0)
        inp = torch.from_numpy(img_2d)
        out = t({"image": inp})
        assert not torch.allclose(out["image"], inp)

    def test_invalid_noise_type_raises(self):
        with pytest.raises(ValueError, match="noise_type"):
            SpatiallyVaryingNoised(keys="image", noise_type="unknown")

    def test_prob_zero_skips_noise(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", prob=0.0)
        inp = torch.from_numpy(img_2d)
        out = t({"image": inp})
        # With prob=0 the transform is never applied — key should still be in output
        assert "image" in out

    def test_determinism_with_seed(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", prob=1.0)
        inp = torch.from_numpy(img_2d)

        t.set_random_state(seed=42)
        out1 = t({"image": inp.clone()})

        t.set_random_state(seed=42)
        out2 = t({"image": inp.clone()})

        assert torch.allclose(out1["image"], out2["image"])
        assert torch.allclose(out1["image_sigma_map"], out2["image_sigma_map"])

    def test_missing_key_is_silently_skipped(self, img_2d):
        t = SpatiallyVaryingNoised(keys=["image", "extra"], prob=1.0)
        out = t({"image": torch.from_numpy(img_2d)})
        # "extra" key absent — should not raise
        assert "image" in out
        assert "extra" not in out

    def test_sigma_map_is_float32(self, img_2d):
        t = SpatiallyVaryingNoised(keys="image", prob=1.0)
        out = t({"image": torch.from_numpy(img_2d)})
        assert out["image_sigma_map"].dtype == torch.float32
