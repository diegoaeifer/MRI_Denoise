"""
Tests for rep2rep and dwi2dwi training modes.

TDD: these tests define the expected behaviour of Rep2RepDataset,
DWI2DWIDataset, and the PerpendicularLoss bridge in CompositeLoss.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti_3d(path, shape=(32, 32, 1)):
    """Save a random 3-D NIfTI file at *path*."""
    import nibabel as nib

    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


def _make_nifti_4d(path, shape=(32, 32, 10, 4)):
    """Save a random 4-D NIfTI file at *path*."""
    import nibabel as nib

    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# Rep2RepDataset
# ---------------------------------------------------------------------------

class TestRep2RepDataset:
    """Behavioural tests for Rep2RepDataset."""

    def _import(self):
        from src.data.dataset import Rep2RepDataset
        return Rep2RepDataset

    def test_rep2rep_dataset_pairs_nifti(self, tmp_path):
        """
        Create two NIfTI files in a subject sub-directory.
        Dataset must find them and return a valid pair.
        """
        Rep2RepDataset = self._import()

        subject_dir = tmp_path / "subject_001"
        subject_dir.mkdir()
        _make_nifti_3d(subject_dir / "rep_1.nii.gz", shape=(32, 32, 1))
        _make_nifti_3d(subject_dir / "rep_2.nii.gz", shape=(32, 32, 1))

        dataset = Rep2RepDataset(tmp_path)

        assert len(dataset) > 0, "Expected at least one pair."
        item = dataset[0]
        assert item is not None, "__getitem__ returned None — loading failed."
        noisy, clean = item
        assert noisy.shape == clean.shape, "noisy and clean shapes must match."
        assert noisy.shape[0] == 1, f"Expected channel dim = 1, got {noisy.shape[0]}."
        assert noisy.ndim == 3, f"Expected (1, H, W), got {noisy.shape}."
        assert noisy.dtype == torch.float32
        assert clean.dtype == torch.float32

    def test_rep2rep_dataset_multi_subject(self, tmp_path):
        """Multiple subject dirs each with 2 reps contribute separate pairs."""
        Rep2RepDataset = self._import()

        for sid in ("subject_001", "subject_002", "subject_003"):
            d = tmp_path / sid
            d.mkdir()
            _make_nifti_3d(d / "rep_1.nii.gz", shape=(16, 16, 1))
            _make_nifti_3d(d / "rep_2.nii.gz", shape=(16, 16, 1))

        dataset = Rep2RepDataset(tmp_path)
        assert len(dataset) == 3, f"Expected 3 pairs, got {len(dataset)}."

    def test_rep2rep_dataset_more_than_two_reps(self, tmp_path):
        """A subject dir with 3 reps yields 2 consecutive pairs."""
        Rep2RepDataset = self._import()

        subject_dir = tmp_path / "subject_001"
        subject_dir.mkdir()
        for i in range(1, 4):
            _make_nifti_3d(subject_dir / f"rep_{i}.nii.gz", shape=(16, 16, 1))

        dataset = Rep2RepDataset(tmp_path)
        assert len(dataset) == 2, (
            f"3 reps should give 2 consecutive pairs, got {len(dataset)}."
        )

    def test_rep2rep_dataset_skips_single_rep_subject(self, tmp_path):
        """A subject dir with only 1 file is skipped gracefully."""
        Rep2RepDataset = self._import()

        subject_dir = tmp_path / "subject_001"
        subject_dir.mkdir()
        _make_nifti_3d(subject_dir / "rep_1.nii.gz", shape=(16, 16, 1))

        dataset = Rep2RepDataset(tmp_path)
        assert len(dataset) == 0, "Single-rep subject should produce no pairs."

    def test_rep2rep_dataset_missing_dir_raises(self):
        """Instantiating with a non-existent directory raises FileNotFoundError."""
        Rep2RepDataset = self._import()

        with pytest.raises(FileNotFoundError):
            Rep2RepDataset("/this/path/does/not/exist")

    def test_rep2rep_tensor_range(self, tmp_path):
        """Returned tensors should be in [0, 1]."""
        Rep2RepDataset = self._import()

        subject_dir = tmp_path / "subject_001"
        subject_dir.mkdir()
        _make_nifti_3d(subject_dir / "rep_1.nii.gz", shape=(32, 32, 1))
        _make_nifti_3d(subject_dir / "rep_2.nii.gz", shape=(32, 32, 1))

        dataset = Rep2RepDataset(tmp_path)
        noisy, clean = dataset[0]
        assert noisy.min().item() >= -1e-6, "noisy below 0."
        assert noisy.max().item() <= 1.0 + 1e-6, "noisy above 1."
        assert clean.min().item() >= -1e-6, "clean below 0."
        assert clean.max().item() <= 1.0 + 1e-6, "clean above 1."


# ---------------------------------------------------------------------------
# DWI2DWIDataset
# ---------------------------------------------------------------------------

class TestDWI2DWIDataset:
    """Behavioural tests for DWI2DWIDataset."""

    def _import(self):
        from src.data.dataset import DWI2DWIDataset
        return DWI2DWIDataset

    def test_dwi2dwi_dataset_flat_ixi_layout(self, tmp_path):
        """
        Flat IXI-DTI layout: per-direction 3-D NIfTI files sharing a subject
        prefix.  Dataset must discover all unique direction pairs.
        """
        DWI2DWIDataset = self._import()

        # 4 direction files for one subject → C(4,2) = 6 pairs
        for i in range(4):
            _make_nifti_3d(
                tmp_path / f"IXI002-Guys-0828-DTI-0{i}.nii.gz",
                shape=(32, 32, 10),
            )

        dataset = DWI2DWIDataset(tmp_path)
        assert len(dataset) > 0, "Expected at least one direction pair."
        item = dataset[0]
        assert item is not None, "__getitem__ returned None."
        a, b = item
        assert a.shape == b.shape, "Direction slices must have identical shape."
        assert a.shape[0] == 1, f"Expected channel dim = 1, got {a.shape[0]}."
        assert a.ndim == 3, f"Expected (1, H, W), got {a.shape}."
        assert a.dtype == torch.float32

    def test_dwi2dwi_dataset_4d_subdir_layout(self, tmp_path):
        """
        Sub-directory layout: each subdir has a 4-D NIfTI.
        Dataset must enumerate volume pairs along the last axis.
        """
        DWI2DWIDataset = self._import()

        subject_dir = tmp_path / "subject_001"
        subject_dir.mkdir()
        # 4 directions: C(4,2) = 6 pairs
        _make_nifti_4d(subject_dir / "dwi.nii.gz", shape=(32, 32, 10, 4))

        dataset = DWI2DWIDataset(tmp_path)
        assert len(dataset) == 6, f"Expected 6 pairs, got {len(dataset)}."
        a, b = dataset[0]
        assert a.shape == b.shape

    def test_dwi2dwi_dataset_missing_dir_raises(self):
        """Non-existent data_dir raises FileNotFoundError."""
        DWI2DWIDataset = self._import()

        with pytest.raises(FileNotFoundError):
            DWI2DWIDataset("/this/path/does/not/exist")

    def test_dwi2dwi_dataset_normalized_range(self, tmp_path):
        """
        Each volume is normalised to [0, 1] independently.
        Returned slices must be in [0, 1].
        """
        DWI2DWIDataset = self._import()

        # Use large random values to exercise normalisation
        import nibabel as nib

        data = (np.random.rand(32, 32, 10) * 1000).astype(np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(img, str(tmp_path / "IXI002-Guys-0828-DTI-00.nii.gz"))

        data2 = (np.random.rand(32, 32, 10) * 500 + 200).astype(np.float32)
        img2 = nib.Nifti1Image(data2, affine=np.eye(4))
        nib.save(img2, str(tmp_path / "IXI002-Guys-0828-DTI-01.nii.gz"))

        dataset = DWI2DWIDataset(tmp_path)
        a, b = dataset[0]
        assert a.min().item() >= -1e-6, "slice_a below 0."
        assert a.max().item() <= 1.0 + 1e-6, "slice_a above 1."
        assert b.min().item() >= -1e-6, "slice_b below 0."
        assert b.max().item() <= 1.0 + 1e-6, "slice_b above 1."

    def test_dwi2dwi_dataset_slice_axis(self, tmp_path):
        """
        slice_axis parameter selects which spatial axis to slice.
        Result shape should match H or W depending on chosen axis.
        """
        DWI2DWIDataset = self._import()

        _make_nifti_3d(
            tmp_path / "IXI002-Guys-0828-DTI-00.nii.gz", shape=(16, 24, 10)
        )
        _make_nifti_3d(
            tmp_path / "IXI002-Guys-0828-DTI-01.nii.gz", shape=(16, 24, 10)
        )

        for axis, expected_hw in [(0, (24, 10)), (1, (16, 10)), (2, (16, 24))]:
            dataset = DWI2DWIDataset(tmp_path, slice_axis=axis)
            a, _ = dataset[0]
            assert a.shape[1:] == torch.Size(expected_hw), (
                f"slice_axis={axis}: expected (1, {expected_hw}), got {a.shape}."
            )

    def test_dwi2dwi_multiple_subjects_flat_layout(self, tmp_path):
        """
        Multiple subjects in flat IXI-DTI layout are handled independently.
        """
        DWI2DWIDataset = self._import()

        for subject_prefix in ("IXI002-Guys-0828", "IXI012-HH-1211"):
            for i in range(3):
                _make_nifti_3d(
                    tmp_path / f"{subject_prefix}-DTI-0{i}.nii.gz",
                    shape=(16, 16, 8),
                )

        dataset = DWI2DWIDataset(tmp_path)
        # Each subject: C(3,2) = 3 pairs → 2 subjects × 3 = 6 total
        assert len(dataset) == 6, f"Expected 6 pairs, got {len(dataset)}."


# ---------------------------------------------------------------------------
# PerpendicularLoss bridge
# ---------------------------------------------------------------------------

class TestPerpendicularLossBridge:
    """Tests for the PerpendicularLoss import bridge in composite.py."""

    def test_perpendicular_loss_flag_exists(self):
        """
        HAS_PERPENDICULAR_LOSS must be a bool regardless of whether WCRR is
        installed.
        """
        from src.losses.composite import HAS_PERPENDICULAR_LOSS

        assert isinstance(HAS_PERPENDICULAR_LOSS, bool), (
            "HAS_PERPENDICULAR_LOSS must be a bool."
        )

    def test_perpendicular_loss_importable_when_wcrr_present(self):
        """
        When WCRR is present HAS_PERPENDICULAR_LOSS should be True and
        _PerpendicularLoss should be the actual class.
        """
        from src.losses import composite as composite_mod

        if not composite_mod.HAS_PERPENDICULAR_LOSS:
            pytest.skip("WCRR project not available in this environment.")

        assert composite_mod._PerpendicularLoss is not None
        # Smoke-test: should be instantiable
        loss_fn = composite_mod._PerpendicularLoss(alpha=0.1)
        x = torch.randn(2, 2, 8, 8)
        y = torch.randn(2, 2, 8, 8)
        result = loss_fn(x, y)
        assert result.ndim == 0, "PerpendicularLoss must return a scalar."

    def test_composite_loss_perpendicular_weight_zero_skips_gracefully(self):
        """
        CompositeLoss with perpendicular weight = 0 must initialise without
        error even when WCRR is absent.
        """
        from src.losses.composite import CompositeLoss

        config = {
            "weights": {
                "l1": 1.0,
                "ssim": 0.1,
                "perpendicular": 0.0,
            },
            "auxiliary": {},
            "data": {"is_3d": False},
        }
        loss = CompositeLoss(config)
        assert loss.perpendicular is None

    def test_composite_loss_perpendicular_weight_nonzero_when_available(self):
        """
        CompositeLoss with perpendicular weight > 0 instantiates
        _PerpendicularLoss when WCRR is available, or logs a warning when not.
        """
        from src.losses import composite as composite_mod
        from src.losses.composite import CompositeLoss

        if not composite_mod.HAS_PERPENDICULAR_LOSS:
            pytest.skip("WCRR project not available — cannot test nonzero weight.")

        config = {
            "weights": {
                "l1": 1.0,
                "ssim": 0.1,
                "perpendicular": 0.5,
            },
            "auxiliary": {
                "perpendicular": {"alpha": 0.1, "eps": 1e-8},
            },
            "data": {"is_3d": False},
        }
        loss = CompositeLoss(config)
        assert loss.perpendicular is not None
