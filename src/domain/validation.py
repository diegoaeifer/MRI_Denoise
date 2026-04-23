import os
from typing import Tuple, Optional
import torch


class InvalidTensorShape(ValueError):
    """Raised when tensor shape doesn't match expected 2-channel format."""
    pass


class InvalidSigmaMap(ValueError):
    """Raised when sigma map values are outside valid range."""
    pass


class InvalidDICOM(ValueError):
    """Raised when DICOM file is corrupted or invalid."""
    pass


def validate_input_tensor(
    x: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """Validate input tensor follows 2-channel convention.

    Args:
        x: Input tensor, expected shape (B, 2, H, W) or (2, H, W)
        expected_shape: Tuple of expected shape. If None, infers from x.

    Returns:
        True if valid.

    Raises:
        InvalidTensorShape: If tensor doesn't match 2-channel format.
    """
    # Check that tensor has at least 3 dimensions: (..., 2, H, W)
    if x.ndim < 3:
        raise InvalidTensorShape(
            f"Expected at least 3D tensor, got shape {x.shape}"
        )

    # Check that channel dimension (position -3) is 2
    if x.shape[-3] != 2:
        raise InvalidTensorShape(
            f"Expected 2 channels at dim -3, got shape {x.shape}. "
            f"Format should be (..., 2, H, W)"
        )

    # If expected_shape provided, validate exact shape
    if expected_shape is not None:
        if x.shape != expected_shape:
            raise InvalidTensorShape(
                f"Expected shape {expected_shape}, got {x.shape}"
            )

    return True


def validate_sigma_map(sigma: torch.Tensor) -> bool:
    """Validate sigma map (noise level map) values.

    Args:
        sigma: Noise level map, expected values in [0, 1]

    Returns:
        True if valid.

    Raises:
        InvalidSigmaMap: If sigma values outside [0, 1] range.
    """
    min_val = sigma.min().item()
    max_val = sigma.max().item()

    if min_val < 0 or max_val > 1:
        raise InvalidSigmaMap(
            f"Sigma map values must be in [0, 1], got min={min_val:.4f}, max={max_val:.4f}"
        )

    return True


def validate_dicom_input(dicom_path: str) -> bool:
    """Validate DICOM file exists and is readable.

    Args:
        dicom_path: Path to DICOM file

    Returns:
        True if valid.

    Raises:
        InvalidDICOM: If file doesn't exist or isn't readable.
    """
    if not os.path.exists(dicom_path):
        raise InvalidDICOM(f"DICOM file not found: {dicom_path}")

    if not os.path.isfile(dicom_path):
        raise InvalidDICOM(f"Path is not a file: {dicom_path}")

    # Try to read first few bytes to check if file is readable
    try:
        with open(dicom_path, "rb") as f:
            _ = f.read(132)  # DICOM header is 132 bytes
    except (IOError, OSError) as e:
        raise InvalidDICOM(f"Cannot read DICOM file {dicom_path}: {str(e)}")

    return True
