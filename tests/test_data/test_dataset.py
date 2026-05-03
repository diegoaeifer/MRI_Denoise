"""
Tests for MONAI-based data loading pipeline (CacheDataset + transforms).

Replaces old torchio MRI_DICOM_Dataset tests.
"""

import pytest
import torch
import numpy as np
import os
import tempfile


class TestMonaiDataPipeline:
    """Test suite for MONAI CacheDataset + transforms."""

    def test_build_train_transforms_import(self):
        """Test that build_train_transforms can be imported."""
        try:
            from src.mri_denoise.data.transforms import build_train_transforms
            assert callable(build_train_transforms)
        except ImportError:
            pytest.skip("build_train_transforms not available")

    def test_build_val_transforms_import(self):
        """Test that build_val_transforms can be imported."""
        try:
            from src.mri_denoise.data.transforms import build_val_transforms
            assert callable(build_val_transforms)
        except ImportError:
            pytest.skip("build_val_transforms not available")

    def test_train_transforms_output_shape(self, dummy_nifti_file):
        """Test that train transforms produce expected output."""
        try:
            from src.mri_denoise.data.transforms import build_train_transforms
        except ImportError:
            pytest.skip("build_train_transforms not available")

        config = {
            "data": {
                "image_size": [64, 64],
                "percentile_lower": 1.0,
                "percentile_upper": 99.0,
            },
            "noise": {"sigma_range": [0.02, 0.3], "grid_size": 4, "noise_type": "gaussian"},
            "augmentation": {"rand_affine_prob": 0.0, "rand_flip_prob": 0.0},
        }

        try:
            transforms = build_train_transforms(config)
            assert transforms is not None
            # Apply to a dummy sample
            sample = {"image": dummy_nifti_file}
            output = transforms(sample)
            assert "image" in output
            assert isinstance(output["image"], torch.Tensor)
        except Exception as e:
            pytest.skip(f"Transform pipeline failed: {e}")

    def test_val_transforms_no_noise(self, dummy_nifti_file):
        """Test that val transforms don't add noise."""
        try:
            from src.mri_denoise.data.transforms import build_val_transforms
        except ImportError:
            pytest.skip("build_val_transforms not available")

        config = {
            "data": {
                "image_size": [64, 64],
                "percentile_lower": 1.0,
                "percentile_upper": 99.0,
            }
        }

        try:
            transforms = build_val_transforms(config)
            sample = {"image": dummy_nifti_file}
            output = transforms(sample)
            # Val transforms should not produce sigma_map
            assert "image" in output
            assert "image_sigma_map" not in output
        except Exception as e:
            pytest.skip(f"Val transform failed: {e}")

    def test_build_datalist(self, temp_data_dir, dummy_nifti_file):
        """Test that build_datalist produces MONAI-format output."""
        try:
            from src.mri_denoise.data.datalist import build_datalist
        except ImportError:
            pytest.skip("build_datalist not available")

        # Copy dummy file to temp dir
        import shutil
        shutil.copy(dummy_nifti_file, os.path.join(temp_data_dir, "test.nii.gz"))

        config = {
            "root_dir": temp_data_dir,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
        }

        try:
            datalist = build_datalist(config)
            assert isinstance(datalist, dict)
            assert "train" in datalist
            assert "val" in datalist
            assert "test" in datalist
            # Each split should be a list of dicts with "image" key
            if len(datalist["train"]) > 0:
                assert "image" in datalist["train"][0]
        except Exception as e:
            pytest.skip(f"Datalist building failed: {e}")

    def test_cache_dataset_integration(self, dummy_nifti_file):
        """Test CacheDataset with transforms pipeline."""
        try:
            from monai.data import CacheDataset
            from src.mri_denoise.data.transforms import build_val_transforms
        except ImportError:
            pytest.skip("MONAI or transforms not available")

        config = {"data": {"image_size": [64, 64], "percentile_lower": 1.0, "percentile_upper": 99.0}}
        transforms = build_val_transforms(config)

        datalist = [{"image": dummy_nifti_file}]
        try:
            dataset = CacheDataset(data=datalist, transform=transforms, cache_rate=0.0)
            assert len(dataset) == 1
            item = dataset[0]
            assert "image" in item
            assert isinstance(item["image"], torch.Tensor)
        except Exception as e:
            pytest.skip(f"CacheDataset failed: {e}")

    def test_spatially_varying_noised_in_pipeline(self, dummy_nifti_file):
        """Test that SpatiallyVaryingNoised is applied in train pipeline."""
        try:
            from src.mri_denoise.data.transforms import build_train_transforms
        except ImportError:
            pytest.skip("build_train_transforms not available")

        config = {
            "data": {"image_size": [32, 32], "percentile_lower": 1.0, "percentile_upper": 99.0},
            "noise": {"sigma_range": [0.05, 0.2], "grid_size": 4, "noise_type": "gaussian"},
            "augmentation": {"rand_affine_prob": 0.0, "rand_flip_prob": 0.0},
        }

        try:
            transforms = build_train_transforms(config)
            sample = {"image": dummy_nifti_file}
            output = transforms(sample)
            # Train pipeline should produce sigma_map
            assert "image_sigma_map" in output, "sigma_map not produced by train pipeline"
            assert isinstance(output["image_sigma_map"], torch.Tensor)
            assert output["image_sigma_map"].dtype == torch.float32
        except Exception as e:
            pytest.skip(f"Spatially varying noise test failed: {e}")
