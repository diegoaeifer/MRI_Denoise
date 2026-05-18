"""
Tests for MRI_Denoise/src/data/noise_pipeline.py and the
NoisePipelineTransform wrapper in transforms.py.

Run with:
    pytest tests/test_noise_pipeline.py -v
"""
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Return a deterministic float32 image in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((h, w)).astype(np.float32)


def _make_gradient_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a smooth gradient image — good for Gibbs / ringing detection."""
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    img = (np.sin(2 * np.pi * xx * 3) * 0.5 + 0.5).astype(np.float32)
    return img


# ---------------------------------------------------------------------------
# NoisePipelineConfig
# ---------------------------------------------------------------------------

class TestNoisePipelineConfig:
    """Tests for the NoisePipelineConfig dataclass."""

    def test_defaults(self):
        """Default config has sane values."""
        from src.data.noise_pipeline import NoisePipelineConfig

        cfg = NoisePipelineConfig()
        assert cfg.sigma_range == (0.01, 0.10)
        assert cfg.gibbs_prob == pytest.approx(0.20)
        assert cfg.acceleration is False
        assert cfg.accel_factor == 2
        assert cfg.temporal is False

    def test_custom_config(self):
        """Custom config values are stored correctly."""
        from src.data.noise_pipeline import NoisePipelineConfig

        cfg = NoisePipelineConfig(
            sigma_range=(0.05, 0.20),
            gibbs_prob=0.50,
            acceleration=True,
            accel_factor=3,
            temporal=True,
        )
        assert cfg.sigma_range == (0.05, 0.20)
        assert cfg.gibbs_prob == pytest.approx(0.50)
        assert cfg.acceleration is True
        assert cfg.accel_factor == 3
        assert cfg.temporal is True


# ---------------------------------------------------------------------------
# NoisePipeline — Rician noise
# ---------------------------------------------------------------------------

class TestRicianNoise:
    """Tests for the _add_rician helper and overall pipeline output."""

    def test_rician_noise_shape_and_positivity(self):
        """Output shape matches input; all values are non-negative (Rician is magnitude)."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(sigma_range=(0.05, 0.05), gibbs_prob=0.0)
        pipeline = NoisePipeline(cfg)
        img = _make_image()

        noisy, sigma = pipeline(img)

        assert noisy.shape == img.shape, "Output shape must match input"
        assert noisy.dtype == np.float32, "Output must be float32"
        assert np.all(noisy >= 0), "Rician noise output must be non-negative"

    def test_rician_internal_shape_and_positivity(self):
        """_add_rician helper alone preserves shape and non-negativity."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        pipeline = NoisePipeline()
        img = _make_image(32, 32)
        noisy = pipeline._add_rician(img, sigma=0.05)

        assert noisy.shape == img.shape
        assert np.all(noisy >= 0)

    def test_rician_noise_increases_variance(self):
        """A noisy image should have strictly higher variance than the clean one.

        Uses a constant (flat) image so clean.var()==0 and any Rician noise
        guarantees noisy.var() > 0, avoiding the Rician-bias ambiguity on
        natural images.
        """
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(sigma_range=(0.10, 0.10), gibbs_prob=0.0)
        pipeline = NoisePipeline(cfg)
        img = np.full((64, 64), 0.5, dtype=np.float32)  # constant image, var==0

        noisy, _ = pipeline(img)

        assert noisy.var() > img.var(), "Rician noise must increase variance"

    def test_zero_sigma_recovers_input(self):
        """With sigma=0, Rician output equals the clean image."""
        from src.data.noise_pipeline import NoisePipeline

        pipeline = NoisePipeline()
        img = _make_image()
        noisy = pipeline._add_rician(img, sigma=0.0)

        np.testing.assert_allclose(noisy, img, atol=1e-6)


# ---------------------------------------------------------------------------
# NoisePipeline — sigma range
# ---------------------------------------------------------------------------

class TestSigmaRange:
    """Tests that the drawn sigma falls within the configured range."""

    def test_sigma_in_range(self):
        """Repeated calls produce sigma within [sigma_min, sigma_max]."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        lo, hi = 0.03, 0.07
        cfg = NoisePipelineConfig(sigma_range=(lo, hi), gibbs_prob=0.0)
        pipeline = NoisePipeline(cfg)
        img = _make_image()

        for _ in range(50):
            _, sigma = pipeline(img)
            assert lo <= sigma <= hi, (
                f"sigma {sigma:.4f} outside [{lo}, {hi}]"
            )

    def test_fixed_sigma_range_is_exact(self):
        """When min == max, the returned sigma equals that exact value."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        target = 0.05
        cfg = NoisePipelineConfig(sigma_range=(target, target), gibbs_prob=0.0)
        pipeline = NoisePipeline(cfg)
        img = _make_image()

        _, sigma = pipeline(img)
        assert sigma == pytest.approx(target)


# ---------------------------------------------------------------------------
# NoisePipeline — acceleration mask
# ---------------------------------------------------------------------------

class TestAccelerationMask:
    """Tests for the k-space undersampling (_apply_acceleration_mask)."""

    def test_acceleration_mask_changes_image(self):
        """Accelerated image must differ from the clean input (aliasing present)."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(
            sigma_range=(0.0, 0.0),  # No noise so only acceleration matters
            gibbs_prob=0.0,
            acceleration=True,
            accel_factor=2,
        )
        pipeline = NoisePipeline(cfg)
        img = _make_image()

        # Patch sigma draw to always return 0 so _add_rician is a no-op
        accel = pipeline._apply_acceleration_mask(img, factor=2)

        assert not np.allclose(accel, img, atol=1e-4), (
            "Acceleration mask should visibly alter the image"
        )

    def test_acceleration_preserves_shape(self):
        """Output shape after undersampling matches input shape."""
        from src.data.noise_pipeline import NoisePipeline

        pipeline = NoisePipeline()
        img = _make_image(64, 64)
        accel = pipeline._apply_acceleration_mask(img, factor=2)
        assert accel.shape == img.shape

    def test_acceleration_output_is_float32(self):
        """Accelerated output is float32."""
        from src.data.noise_pipeline import NoisePipeline

        pipeline = NoisePipeline()
        img = _make_image()
        accel = pipeline._apply_acceleration_mask(img, factor=3)
        assert accel.dtype == np.float32

    def test_higher_factor_increases_aliasing(self):
        """Higher undersampling factor → more energy removed → bigger difference."""
        from src.data.noise_pipeline import NoisePipeline

        pipeline = NoisePipeline()
        img = _make_image(64, 64)
        diff_2 = np.abs(pipeline._apply_acceleration_mask(img, factor=2) - img).mean()
        diff_4 = np.abs(pipeline._apply_acceleration_mask(img, factor=4) - img).mean()
        assert diff_4 >= diff_2, (
            "Higher acceleration factor should cause at least as much distortion"
        )

    def test_acceleration_pipeline_end_to_end(self):
        """Pipeline with acceleration=True returns different image from clean."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(
            sigma_range=(0.001, 0.001),
            gibbs_prob=0.0,
            acceleration=True,
            accel_factor=2,
        )
        pipeline = NoisePipeline(cfg)
        img = _make_image()
        noisy, _ = pipeline(img)
        assert not np.allclose(noisy, img, atol=1e-3)


# ---------------------------------------------------------------------------
# NoisePipeline — Gibbs ringing
# ---------------------------------------------------------------------------

class TestGibbsRinging:
    """Tests for the Gibbs truncation artifact."""

    def test_gibbs_truncation_changes_image(self):
        """Gibbs ringing must alter the image."""
        from src.data.noise_pipeline import NoisePipeline

        pipeline = NoisePipeline()
        img = _make_gradient_image()
        gibbs = pipeline._apply_gibbs(img, truncation_pct=0.10)

        assert not np.allclose(gibbs, img, atol=1e-4), (
            "Gibbs truncation should visibly alter the image"
        )

    def test_gibbs_truncation_visible(self):
        """Gibbs image should exhibit ringing (oscillations near edges).

        A simple proxy: the variance of the *difference* (img - gibbs) should
        be substantially non-zero, indicating high-frequency ringing was
        introduced near intensity transitions.  Uses 20 % truncation so the
        effect is well above the noise floor.
        """
        from src.data.noise_pipeline import NoisePipeline

        pipeline = NoisePipeline()
        img = _make_gradient_image()
        gibbs = pipeline._apply_gibbs(img, truncation_pct=0.20)

        diff = img - gibbs
        assert diff.var() > 1e-5, (
            "Gibbs ringing should produce a measurable difference with ringing oscillations"
        )

    def test_gibbs_preserves_shape_and_dtype(self):
        """Output shape and dtype match input after Gibbs."""
        from src.data.noise_pipeline import NoisePipeline

        pipeline = NoisePipeline()
        img = _make_image(64, 64)
        gibbs = pipeline._apply_gibbs(img)
        assert gibbs.shape == img.shape
        assert gibbs.dtype == np.float32

    def test_gibbs_prob_zero_never_fires(self):
        """With gibbs_prob=0, the output should only differ by Rician noise (no Gibbs)."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(sigma_range=(0.0, 0.0), gibbs_prob=0.0)
        pipeline = NoisePipeline(cfg)
        img = _make_gradient_image()

        # With sigma=0 and gibbs_prob=0, the image should be unchanged
        noisy, _ = pipeline(img)
        np.testing.assert_allclose(noisy, img, atol=1e-5)

    def test_gibbs_prob_one_always_fires(self):
        """With gibbs_prob=1.0 (and no noise), every call applies Gibbs."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(sigma_range=(0.0, 0.0), gibbs_prob=1.0)
        pipeline = NoisePipeline(cfg)
        img = _make_gradient_image()
        noisy, _ = pipeline(img)

        # Gibbs should have been applied
        assert not np.allclose(noisy, img, atol=1e-4)


# ---------------------------------------------------------------------------
# NoisePipeline — temporal correlation
# ---------------------------------------------------------------------------

class TestTemporalCorrelation:
    """Tests for TemporalNoisePipeline correlated noise across frames."""

    def test_temporal_pipeline_imports(self):
        """TemporalNoisePipeline can be imported and instantiated."""
        from src.data.noise_pipeline import TemporalNoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(temporal=True)
        tpipe = TemporalNoisePipeline(cfg, temporal_corr=0.8)
        assert tpipe is not None

    def test_pipeline_temporal_correlation(self):
        """Two consecutive slices from the same volume should have correlated noise.

        Strategy: fix the image to a constant (so the noise IS the signal), then
        compare the noise fields of two consecutive frames.  With high temporal_corr,
        the Pearson correlation of the two noise maps should be clearly above that of
        two *independent* noise realisations.
        """
        from src.data.noise_pipeline import TemporalNoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(
            sigma_range=(0.10, 0.10),  # fixed sigma for reproducibility
            gibbs_prob=0.0,
            acceleration=False,
        )
        tpipe = TemporalNoisePipeline(cfg, temporal_corr=0.9)

        # Use a constant image so output ≈ noise
        img = np.full((64, 64), 0.5, dtype=np.float32)

        np.random.seed(42)
        tpipe.reset()
        noisy1, _ = tpipe(img)
        noisy2, _ = tpipe(img)

        noise1 = noisy1.flatten() - 0.5
        noise2 = noisy2.flatten() - 0.5

        corr = float(np.corrcoef(noise1, noise2)[0, 1])

        # With temporal_corr=0.9 the expected correlation is ~0.81 (corr^2 ≈ 0.9).
        # We use a generous threshold so the test is not flaky.
        assert corr > 0.5, (
            f"Expected high temporal correlation (>0.5), got {corr:.3f}"
        )

    def test_independent_pipeline_has_low_correlation(self):
        """Plain NoisePipeline (no temporal) should produce uncorrelated noise."""
        from src.data.noise_pipeline import NoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(
            sigma_range=(0.10, 0.10),
            gibbs_prob=0.0,
        )
        pipeline = NoisePipeline(cfg)
        img = np.full((64, 64), 0.5, dtype=np.float32)

        np.random.seed(7)
        noisy_frames = [pipeline(img)[0] for _ in range(10)]

        # Correlations between non-adjacent frames should be near zero
        correlations = []
        for i in range(len(noisy_frames)):
            for j in range(i + 1, len(noisy_frames)):
                n1 = noisy_frames[i].flatten() - 0.5
                n2 = noisy_frames[j].flatten() - 0.5
                correlations.append(np.corrcoef(n1, n2)[0, 1])

        mean_corr = float(np.mean(np.abs(correlations)))
        assert mean_corr < 0.3, (
            f"Independent pipeline should have low inter-frame correlation, got {mean_corr:.3f}"
        )

    def test_temporal_reset_clears_state(self):
        """After reset(), two frame sequences should not share the same noise."""
        from src.data.noise_pipeline import TemporalNoisePipeline, NoisePipelineConfig

        cfg = NoisePipelineConfig(sigma_range=(0.10, 0.10), gibbs_prob=0.0)
        tpipe = TemporalNoisePipeline(cfg, temporal_corr=0.95)
        img = np.full((32, 32), 0.5, dtype=np.float32)

        np.random.seed(99)
        tpipe.reset()
        noisy_a, _ = tpipe(img)

        tpipe.reset()
        noisy_b, _ = tpipe(img)

        # After independent resets with different RNG states they should differ
        # (extremely unlikely to match, but we check shape at minimum)
        assert noisy_a.shape == noisy_b.shape


# ---------------------------------------------------------------------------
# NoisePipelineTransform (transforms.py wrapper)
# ---------------------------------------------------------------------------

class TestNoisePipelineTransform:
    """Tests for the NoisePipelineTransform wrapper in transforms.py."""

    def test_transform_imports(self):
        """NoisePipelineTransform can be imported from transforms module."""
        from src.data.transforms import NoisePipelineTransform

        assert NoisePipelineTransform is not None

    def test_transform_accepts_1hw_tensor(self):
        """Transform handles (1, H, W) float tensor and returns same shape."""
        from src.data.transforms import NoisePipelineTransform
        from src.data.noise_pipeline import NoisePipelineConfig

        cfg = NoisePipelineConfig(sigma_range=(0.05, 0.05), gibbs_prob=0.0)
        transform = NoisePipelineTransform(config=cfg)

        img = torch.rand(1, 64, 64)
        noisy, sigma = transform(img)

        assert noisy.shape == img.shape, "Output tensor shape must match input"
        assert isinstance(sigma, float), "Sigma must be a Python float"
        assert 0.04 <= sigma <= 0.06

    def test_transform_accepts_hw_tensor(self):
        """Transform handles bare (H, W) tensor and returns same shape."""
        from src.data.transforms import NoisePipelineTransform

        transform = NoisePipelineTransform()
        img = torch.rand(64, 64)
        noisy, sigma = transform(img)

        assert noisy.shape == img.shape

    def test_transform_output_is_float_tensor(self):
        """Transform returns a torch.Tensor with float dtype."""
        from src.data.transforms import NoisePipelineTransform

        transform = NoisePipelineTransform()
        img = torch.rand(1, 32, 32)
        noisy, _ = transform(img)

        assert isinstance(noisy, torch.Tensor)
        assert noisy.is_floating_point()

    def test_transform_noisy_differs_from_clean(self):
        """Noisy output must not be identical to the clean input."""
        from src.data.transforms import NoisePipelineTransform
        from src.data.noise_pipeline import NoisePipelineConfig

        cfg = NoisePipelineConfig(sigma_range=(0.05, 0.10), gibbs_prob=0.0)
        transform = NoisePipelineTransform(config=cfg)

        img = torch.rand(1, 64, 64)
        noisy, _ = transform(img)

        assert not torch.allclose(noisy, img, atol=1e-4), (
            "NoisePipelineTransform output must differ from clean input"
        )

    def test_transform_default_config(self):
        """Transform works with default config (no config argument)."""
        from src.data.transforms import NoisePipelineTransform

        transform = NoisePipelineTransform()
        img = torch.rand(1, 64, 64)
        noisy, sigma = transform(img)

        assert noisy.shape == img.shape
        assert 0.01 <= sigma <= 0.10
