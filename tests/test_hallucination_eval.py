"""
Tests for scripts/hallucination_eval.py

TDD: these tests must pass even when the optional DIDSR repos (sfrc, DLMO)
are not installed — all external imports inside the module are guarded by
try/except blocks.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root and the script directory are importable regardless
# of how pytest is invoked.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from hallucination_eval import (  # noqa: E402
    add_synthetic_lesions,
    compute_dlmo_score,
    compute_sfrc_score,
)


# ---------------------------------------------------------------------------
# add_synthetic_lesions
# ---------------------------------------------------------------------------


class TestAddSyntheticLesions:
    def test_shape_preserved(self):
        """Output image must have the same shape as the input."""
        img = np.random.rand(64, 64).astype(np.float32)
        modified, masks = add_synthetic_lesions(img, n_lesions=4)
        assert modified.shape == img.shape

    def test_correct_number_of_masks(self):
        """One mask per lesion must be returned."""
        img = np.random.rand(64, 64).astype(np.float32)
        _modified, masks = add_synthetic_lesions(img, n_lesions=4)
        assert len(masks) == 4

    def test_masks_are_same_shape_as_input(self):
        """Every mask must match the input image shape."""
        img = np.random.rand(64, 64).astype(np.float32)
        _modified, masks = add_synthetic_lesions(img, n_lesions=3)
        for m in masks:
            assert m.shape == img.shape

    def test_modifies_image(self):
        """Lesions must add signal — a zero image gains positive pixels."""
        img = np.zeros((64, 64), dtype=np.float32)
        modified, _ = add_synthetic_lesions(img, n_lesions=1)
        assert modified.sum() > 0

    def test_zero_lesions_returns_unchanged(self):
        """Requesting 0 lesions must return an unchanged image and no masks."""
        img = np.random.rand(64, 64).astype(np.float32)
        modified, masks = add_synthetic_lesions(img, n_lesions=0)
        np.testing.assert_array_equal(modified, img)
        assert len(masks) == 0

    def test_masks_are_binary(self):
        """Disk masks must contain only 0 or 1 values."""
        img = np.random.rand(64, 64).astype(np.float32)
        _modified, masks = add_synthetic_lesions(img, n_lesions=2)
        for m in masks:
            unique = np.unique(m)
            assert set(unique.tolist()).issubset({0.0, 1.0}), (
                f"Mask contains non-binary values: {unique}"
            )

    def test_dtype_preserved(self):
        """Return dtype must match the input dtype."""
        img = np.zeros((64, 64), dtype=np.float64)
        modified, _ = add_synthetic_lesions(img, n_lesions=1)
        assert modified.dtype == img.dtype

    def test_single_lesion_mask_has_nonzero_pixels(self):
        """At least one pixel in each mask must be 1."""
        img = np.zeros((64, 64), dtype=np.float32)
        _modified, masks = add_synthetic_lesions(img, n_lesions=1)
        assert masks[0].sum() > 0


# ---------------------------------------------------------------------------
# compute_sfrc_score
# ---------------------------------------------------------------------------


class TestComputeSfrcScore:
    def test_returns_dict(self):
        """Must always return a dict, even when sfrc is not installed."""
        arr = np.random.rand(64, 64).astype(np.float32)
        result = compute_sfrc_score(arr, arr)
        assert isinstance(result, dict)

    def test_sfrc_score_key_present(self):
        """Return value must contain the 'sfrc_score' key."""
        arr = np.random.rand(64, 64).astype(np.float32)
        result = compute_sfrc_score(arr, arr)
        assert "sfrc_score" in result

    def test_sfrc_score_is_numeric(self):
        """'sfrc_score' must be a float (possibly NaN)."""
        arr = np.random.rand(64, 64).astype(np.float32)
        result = compute_sfrc_score(arr, arr)
        assert isinstance(result["sfrc_score"], float)

    def test_graceful_on_missing_sfrc(self, monkeypatch):
        """If sfrc cannot be imported, score must be NaN and 'error' key present."""
        # Simulate ImportError by temporarily hiding frc_utils from sys.modules
        monkeypatch.setitem(sys.modules, "frc_utils", None)
        arr = np.random.rand(64, 64).astype(np.float32)
        result = compute_sfrc_score(arr, arr)
        assert np.isnan(result["sfrc_score"])
        assert "error" in result

    def test_accepts_3d_input(self):
        """Must handle 3-D arrays by selecting the central slice."""
        arr = np.random.rand(16, 64, 64).astype(np.float32)
        result = compute_sfrc_score(arr, arr)
        assert "sfrc_score" in result

    def test_identical_images_high_score_or_nan(self):
        """
        For identical images the FRC should be close to 1.0 or NaN (library
        absent). Either outcome is acceptable here — we just verify no
        exception is raised and the key exists.
        """
        arr = np.random.rand(64, 64).astype(np.float32)
        result = compute_sfrc_score(arr, arr)
        score = result["sfrc_score"]
        assert np.isnan(score) or 0.0 <= score <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# compute_dlmo_score
# ---------------------------------------------------------------------------


class TestComputeDlmoScore:
    def test_returns_dict(self):
        """Must always return a dict."""
        arr = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = compute_dlmo_score(arr, mask)
        assert isinstance(result, dict)

    def test_lesions_detected_key_present(self):
        """Return value must contain the 'lesions_detected' key."""
        arr = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = compute_dlmo_score(arr, mask)
        assert "lesions_detected" in result

    def test_lesions_detected_is_numeric(self):
        """'lesions_detected' must be a float (possibly NaN)."""
        arr = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = compute_dlmo_score(arr, mask)
        assert isinstance(result["lesions_detected"], float)

    def test_nonzero_mask_produces_scores(self):
        """A non-empty mask must populate the 'lesion_scores' list."""
        arr = np.ones((64, 64), dtype=np.float32) * 0.5
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[28:36, 28:36] = 1.0  # 8×8 lesion patch
        result = compute_dlmo_score(arr, mask)
        assert "lesion_scores" in result
        assert len(result["lesion_scores"]) > 0

    def test_empty_mask_handled_gracefully(self):
        """An all-zero mask must not raise; may return NaN with an 'error' key."""
        arr = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        result = compute_dlmo_score(arr, mask)
        # No exception; the value is either NaN or a float in [0, 1]
        ld = result["lesions_detected"]
        assert np.isnan(ld) or 0.0 <= ld <= 1.0

    def test_accepts_3d_input(self):
        """Must handle 3-D image + 3-D mask by using the central slice."""
        arr = np.random.rand(16, 64, 64).astype(np.float32)
        mask = np.zeros((16, 64, 64), dtype=np.float32)
        result = compute_dlmo_score(arr, mask)
        assert "lesions_detected" in result

    def test_detected_fraction_in_range(self):
        """'lesions_detected' for a non-empty mask must be in [0, 1]."""
        arr = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        result = compute_dlmo_score(arr, mask)
        ld = result["lesions_detected"]
        if not np.isnan(ld):
            assert 0.0 <= ld <= 1.0
