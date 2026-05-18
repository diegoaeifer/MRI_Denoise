"""Tests for EnhancementPipeline."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _gray(h=64, w=64, seed=0):
    return np.random.default_rng(seed).random((h, w), dtype=np.float32)


def test_empty_pipeline_is_identity():
    from enhancement import EnhancementPipeline
    pipe = EnhancementPipeline([])
    img = _gray()
    np.testing.assert_array_equal(pipe.apply(img), img)


def test_unknown_step_raises():
    from enhancement import EnhancementPipeline
    with pytest.raises(ValueError, match="Unknown enhancement step"):
        EnhancementPipeline([{"type": "magic"}])


def test_clahe_output_range():
    from enhancement import EnhancementPipeline
    pipe = EnhancementPipeline([{"type": "clahe", "clip_limit": 0.03, "kernel_size": 8}])
    out = pipe.apply(_gray())
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_unsharp_mask_output_range():
    from enhancement import EnhancementPipeline
    pipe = EnhancementPipeline([{"type": "unsharp_mask", "radius": 1.0, "amount": 1.0}])
    out = pipe.apply(_gray())
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_dither_quantizes_to_levels():
    from enhancement import EnhancementPipeline
    pipe = EnhancementPipeline([{"type": "dither", "bits": 4}])
    out = pipe.apply(_gray())
    assert out.dtype == np.float32
    levels = 2**4
    quantized = np.round(out * (levels - 1)) / (levels - 1)
    np.testing.assert_allclose(out, quantized, atol=1e-5)


def test_interpolate_upsamples():
    from enhancement import EnhancementPipeline
    pipe = EnhancementPipeline([{"type": "interpolate", "scale_factor": 2.0}])
    out = pipe.apply(_gray(32, 32))
    assert out.shape == (64, 64)
    assert out.dtype == np.float32


def test_interpolate_clips_to_01():
    from enhancement import EnhancementPipeline
    pipe = EnhancementPipeline([{"type": "interpolate", "scale_factor": 1.5}])
    out = pipe.apply(np.ones((32, 32), dtype=np.float32))
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_pipeline_chain_shape_preserved():
    from enhancement import EnhancementPipeline
    pipe = EnhancementPipeline([
        {"type": "clahe",        "clip_limit": 0.01},
        {"type": "unsharp_mask", "radius": 0.5, "amount": 0.3},
        {"type": "dither",       "bits": 7},
    ])
    img = _gray()
    out = pipe.apply(img)
    assert out.shape == img.shape
    assert out.dtype == np.float32
