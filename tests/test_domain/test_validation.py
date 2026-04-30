import sys
import os
from unittest.mock import MagicMock, patch, mock_open
import pytest


# --- Mocking Setup ---
# Define MockTensor independently of torch
class MockTensor:
    def __init__(self, shape, min_val=0.5, max_val=0.5):
        self.shape = shape
        self.ndim = len(shape)
        self._min_val = min_val
        self._max_val = max_val

    def min(self):
        m = MagicMock()
        m.item.return_value = self._min_val
        return m

    def max(self):
        m = MagicMock()
        m.item.return_value = self._max_val
        return m

    def __repr__(self):
        return f"MockTensor(shape={self.shape})"


# Only mock torch if it's missing to avoid poisoning environments where it's installed (like CI)
try:
    import torch
except ImportError:
    mock_torch = MagicMock()
    mock_torch.Tensor = MockTensor
    sys.modules["torch"] = mock_torch

# Now we can import the items to test from src.domain.validation
from src.domain.validation import (
    validate_input_tensor,
    validate_sigma_map,
    validate_dicom_input,
    InvalidTensorShape,
    InvalidSigmaMap,
    InvalidDICOM,
)


class TestValidation:
    """Test suite for domain validation logic."""

    # --- Tests for validate_input_tensor ---
    def test_validate_input_tensor_valid_3d(self):
        """Test valid 3D tensor input (2, H, W)."""
        x = MockTensor(shape=(2, 256, 256))
        assert validate_input_tensor(x) is True

    def test_validate_input_tensor_valid_4d(self):
        """Test valid 4D tensor input (B, 2, H, W)."""
        x = MockTensor(shape=(4, 2, 256, 256))
        assert validate_input_tensor(x) is True

    def test_validate_input_tensor_invalid_ndim(self):
        """Test tensor with less than 3 dimensions."""
        x = MockTensor(shape=(2, 256))
        with pytest.raises(InvalidTensorShape, match="Expected at least 3D tensor"):
            validate_input_tensor(x)

    def test_validate_input_tensor_invalid_channels(self):
        """Test tensor with wrong number of channels at dim -3."""
        x = MockTensor(shape=(1, 256, 256))  # Dim -3 is 1, not 2
        with pytest.raises(InvalidTensorShape, match="Expected 2 channels at dim -3"):
            validate_input_tensor(x)

    def test_validate_input_tensor_mismatched_expected_shape(self):
        """Test tensor with shape mismatch against expected_shape."""
        x = MockTensor(shape=(2, 256, 256))
        expected = (2, 128, 128)
        with pytest.raises(InvalidTensorShape, match="Expected shape"):
            validate_input_tensor(x, expected_shape=expected)

    def test_validate_input_tensor_matching_expected_shape(self):
        """Test tensor matching exact expected_shape."""
        shape = (2, 256, 256)
        x = MockTensor(shape=shape)
        assert validate_input_tensor(x, expected_shape=shape) is True

    # --- Tests for validate_sigma_map ---
    def test_validate_sigma_map_valid(self):
        """Test valid sigma map with values in [0, 1]."""
        sigma = MockTensor(shape=(1, 256, 256), min_val=0.0, max_val=1.0)
        assert validate_sigma_map(sigma) is True

    def test_validate_sigma_map_too_low(self):
        """Test sigma map with values below 0."""
        sigma = MockTensor(shape=(1, 256, 256), min_val=-0.1, max_val=0.5)
        with pytest.raises(
            InvalidSigmaMap, match=r"Sigma map values must be in \[0, 1\]"
        ):
            validate_sigma_map(sigma)

    def test_validate_sigma_map_too_high(self):
        """Test sigma map with values above 1."""
        sigma = MockTensor(shape=(1, 256, 256), min_val=0.5, max_val=1.1)
        with pytest.raises(
            InvalidSigmaMap, match=r"Sigma map values must be in \[0, 1\]"
        ):
            validate_sigma_map(sigma)

    # --- Tests for validate_dicom_input ---
    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_validate_dicom_input_success(self, mock_isfile, mock_exists):
        """Test successful DICOM validation."""
        mock_exists.return_value = True
        mock_isfile.return_value = True

        # DICOM standard has 'DICM' prefix starting at offset 128
        dummy_data = b"\0" * 128 + b"DICM"
        m = mock_open(read_data=dummy_data)
        with patch("builtins.open", m):
            assert validate_dicom_input("dummy.dcm") is True

        # Verify it tried to read the DICOM header
        m().read.assert_called_with(132)

    @patch("os.path.exists")
    def test_validate_dicom_input_not_found(self, mock_exists):
        """Test validation fails if file does not exist."""
        mock_exists.return_value = False
        with pytest.raises(InvalidDICOM, match="DICOM file not found"):
            validate_dicom_input("missing.dcm")

    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_validate_dicom_input_is_directory(self, mock_isfile, mock_exists):
        """Test validation fails if path is a directory."""
        mock_exists.return_value = True
        mock_isfile.return_value = False
        with pytest.raises(InvalidDICOM, match="Path is not a file"):
            validate_dicom_input("directory/")

    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_validate_dicom_input_unreadable(self, mock_isfile, mock_exists):
        """Test validation fails if file is unreadable."""
        mock_exists.return_value = True
        mock_isfile.return_value = True

        with patch("builtins.open", mock_open()) as mocked_file:
            mocked_file.side_effect = IOError("Permission denied")
            with pytest.raises(InvalidDICOM, match="Cannot read DICOM file"):
                validate_dicom_input("locked.dcm")
