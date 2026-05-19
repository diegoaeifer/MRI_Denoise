# MRI_Denoise/tests/search/test_dither.py
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from search.dither import DITHER_FNS, apply_dither

RNG = np.random.default_rng(0)
CLEAN = RNG.uniform(0.1, 0.9, (64, 64)).astype(np.float32)
SIGMA = 0.1


def test_none_is_identity():
    out = apply_dither(CLEAN, SIGMA, {"name": "none"})
    np.testing.assert_array_equal(out, CLEAN)


def test_gaussian_changes_image():
    out = apply_dither(CLEAN, SIGMA, {"name": "gaussian", "strength": 0.1})
    assert not np.array_equal(out, CLEAN)


def test_gaussian_clipped_to_0_1():
    out = apply_dither(CLEAN, SIGMA, {"name": "gaussian", "strength": 0.3})
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_blue_changes_image():
    out = apply_dither(CLEAN, SIGMA, {"name": "blue", "strength": 0.1})
    assert not np.array_equal(out, CLEAN)


def test_all_names_in_dither_fns():
    assert set(DITHER_FNS) == {"none", "gaussian", "blue"}


def test_gaussian_psnr_at_small_strength_stays_high():
    from skimage.metrics import peak_signal_noise_ratio
    out = apply_dither(CLEAN, SIGMA, {"name": "gaussian", "strength": 0.02})
    psnr = peak_signal_noise_ratio(CLEAN, out, data_range=1.0)
    assert psnr > 35.0, f"PSNR {psnr:.1f} too low for tiny dither"
