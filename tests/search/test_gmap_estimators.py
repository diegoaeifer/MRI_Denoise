import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from search.gmap_estimators import (
    gmap_uniform, gmap_local_variance, gmap_wavelet,
    gmap_gradient, gmap_mad, GMAP_FNS,
)

RNG = np.random.default_rng(42)
IMG = RNG.uniform(0, 1, (64, 64)).astype(np.float32)


def test_uniform_is_all_ones():
    g = gmap_uniform(IMG)
    assert g.shape == IMG.shape
    np.testing.assert_array_equal(g, np.ones_like(g))


def test_all_strategies_return_float32_same_shape():
    for name, fn in GMAP_FNS.items():
        g = fn(IMG)
        assert g.dtype == np.float32, f"{name}: expected float32"
        assert g.shape == IMG.shape, f"{name}: shape mismatch"


def test_all_strategies_clipped_0p1_to_10():
    for name, fn in GMAP_FNS.items():
        g = fn(IMG)
        assert g.min() >= 0.09, f"{name}: min below 0.1: {g.min()}"
        assert g.max() <= 10.01, f"{name}: max above 10: {g.max()}"


def test_local_variance_not_uniform_on_structured_image():
    structured = np.zeros((64, 64), dtype=np.float32)
    structured[24:40, 24:40] = 1.0
    g = gmap_local_variance(structured, window=7)
    assert g.std() > 0.01, "local_variance gmap should not be uniform on structured input"


def test_gmap_fns_keys():
    expected = {"uniform", "local_var_5", "local_var_9", "wavelet", "gradient", "mad_8"}
    assert set(GMAP_FNS.keys()) == expected
