import os
import sys
import pytest
import tempfile
from unittest.mock import MagicMock, patch

import src.data.check_background
from src.data.check_background import process_batch

def test_process_batch_error_path():
    """
    Tests that process_batch correctly handles and suppresses errors for:
    - Missing files
    - Invalid DICOM files
    - Valid DICOM files that don't have PixelData
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy file paths
        valid_file = os.path.join(tmpdir, "valid.dcm")
        invalid_file = os.path.join(tmpdir, "invalid.dcm")
        missing_file = os.path.join(tmpdir, "missing.dcm")
        no_pixel_file = os.path.join(tmpdir, "no_pixel.dcm")

        # Write some data so the valid and invalid files exist
        with open(valid_file, "wb") as f: f.write(b"valid")
        with open(invalid_file, "wb") as f: f.write(b"invalid")
        with open(no_pixel_file, "wb") as f: f.write(b"no_pixel")

        file_paths = [valid_file, invalid_file, missing_file, no_pixel_file]

        # Mock pydicom.dcmread within the actual module
        with patch('src.data.check_background.pydicom') as mock_pydicom:
            def mock_dcmread(fp, force=True):
                if fp == invalid_file:
                    raise Exception("Invalid DICOM file")
                if fp == missing_file:
                    raise FileNotFoundError("File not found")

                if fp == no_pixel_file:
                    # Valid DICOM, but no PixelData attribute
                    mock_ds = MagicMock()
                    del mock_ds.PixelData
                    return mock_ds

                # Valid file mock
                mock_ds = MagicMock()
                # Must have PixelData attribute
                mock_ds.PixelData = b"some data"

                # mock_ds.pixel_array.astype should return a mock array
                mock_array = MagicMock()
                mock_array.shape = (10, 10)
                mock_array.__getitem__.return_value = [1, 2, 3] # for slicing [::stride, ::stride]
                mock_ds.pixel_array.astype.return_value = mock_array

                return mock_ds

            mock_pydicom.dcmread.side_effect = mock_dcmread

            with patch('src.data.check_background.np') as mock_np:
                # Make np.quantile return fixed values to simulate p1, p99
                mock_np.quantile.return_value = (10.0, 50.0)

                # We expect process_batch to not crash and to process valid_file,
                # while silently skipping invalid_file, missing_file, and no_pixel_file
                problematic = process_batch(file_paths, threshold_range=1000)

                # As long as it didn't crash, the exception handler works
                assert isinstance(problematic, list)

                # Check that only the valid file was processed successfully and returned
                # since dynamic range is 50.0 - 10.0 = 40.0 < 1000
                assert len(problematic) == 1
                assert problematic[0][0] == valid_file
                assert problematic[0][1] == 40.0
