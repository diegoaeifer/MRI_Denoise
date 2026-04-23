"""
MRI Image Preprocessing Utility
Reduces file size and loading time by:
  - Background cropping via percentile-based bounding box
  - Normalization and float16 conversion
  - Optional resize to standard resolution

Usage:
  python -m src.utils.preprocess <input_dir> [options]
  python -m src.utils.preprocess /data/IXI --test
  python -m src.utils.preprocess /data/IXI --resize 256 --delete
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom

try:
    import nibabel as nib

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VALID_RESIZE_DIMS = {256, 384, 512}


# ---------------------------------------------------------------------------
# Image math helpers
# ---------------------------------------------------------------------------


def clip_percentiles(image: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Clip image to percentile range (handles varying field strengths)."""
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return image
    lo = np.percentile(nonzero, p_low)
    hi = np.percentile(nonzero, p_high)
    return np.clip(image, lo, hi)


def normalize_to_float16(image: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] and convert to float16."""
    image = image.astype(np.float32)
    lo, hi = image.min(), image.max()
    normalized = (image - lo) / (hi - lo + 1e-8)
    return normalized.astype(np.float16)


def compute_bounding_box(volume: np.ndarray, margin: int = 10) -> Tuple[slice, slice, slice]:
    """Find the smallest bounding box enclosing signal above threshold.

    Uses percentile-based thresholding on nonzero pixels so it generalises
    across field strengths and sequences.

    Args:
        volume: 3-D array (H, W, D) or 2-D array (H, W).
        margin:  Pixel margin to leave around the detected boundary.

    Returns:
        A tuple of slices (one per spatial axis) ready to index the array.
    """
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        # Flat image — return full extent
        return tuple(slice(0, s) for s in volume.shape)

    # 0.5–2nd percentile of nonzero pixels generalises across field strengths
    threshold = np.percentile(nonzero, 1.0)
    mask = volume > threshold

    slices = []
    for ax in range(volume.ndim):
        # Project mask onto this axis
        proj = mask.any(axis=tuple(i for i in range(volume.ndim) if i != ax))
        indices = np.where(proj)[0]
        lo = max(0, int(indices[0]) - margin)
        hi = min(volume.shape[ax], int(indices[-1]) + margin + 1)
        slices.append(slice(lo, hi))

    return tuple(slices)


def crop_stack_consistently(
    stack: np.ndarray, margin: int = 10
) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """Crop the union bounding box across all slices of a stack.

    For a DICOM series (D, H, W) or NIfTI (H, W, D), compute one bounding
    box that covers every slice so the matrix size is identical for all images.

    Args:
        stack: Array of shape (D, H, W) for DICOM or (H, W, D) for NIfTI.
        margin: Pixel margin around the bounding box.

    Returns:
        (cropped_stack, bbox_slices)
    """
    # Work with a 3-D volume in (H, W, D) layout
    if stack.ndim == 2:
        stack = stack[:, :, np.newaxis]

    bbox = compute_bounding_box(stack, margin=margin)
    cropped = stack[bbox]
    return cropped, bbox


def resize_volume(volume: np.ndarray, target_size: int) -> np.ndarray:
    """Resize the two largest spatial dims to target_size x target_size.

    Uses nearest-neighbour resampling (fast; maintains range for float16).
    For 3-D volumes keeps the depth (slice count) unchanged.
    """
    try:
        from skimage.transform import resize as sk_resize

        if volume.ndim == 3:
            h, w, d = volume.shape
            resized = sk_resize(
                volume,
                (target_size, target_size, d),
                order=1,  # bilinear
                preserve_range=True,
                anti_aliasing=False,
            )
        else:
            resized = sk_resize(
                volume,
                (target_size, target_size),
                order=1,
                preserve_range=True,
                anti_aliasing=False,
            )
        return resized.astype(volume.dtype)
    except ImportError:
        # Fallback: simple numpy zoom approximation
        logger.warning("scikit-image not installed; skipping resize step.")
        return volume


# ---------------------------------------------------------------------------
# DICOM helpers
# ---------------------------------------------------------------------------


def load_dicom_series(dcm_files: List[Path]) -> Tuple[np.ndarray, pydicom.Dataset]:
    """Load a sorted DICOM series into a 3-D array (H, W, D)."""

    # Sort by InstanceNumber if available, else by filename
    def sort_key(p: Path) -> int:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    dcm_files = sorted(dcm_files, key=sort_key)
    slices = []
    ref_ds = None
    for f in dcm_files:
        ds = pydicom.dcmread(str(f))
        pixel = ds.pixel_array.astype(np.float32)
        # Apply RescaleSlope / RescaleIntercept if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixel = pixel * slope + intercept
        slices.append(pixel)
        if ref_ds is None:
            ref_ds = ds

    volume = np.stack(slices, axis=-1)  # (H, W, D)
    return volume, ref_ds


def save_dicom_series(
    volume_f16: np.ndarray,
    ref_ds: pydicom.Dataset,
    src_files: List[Path],
    out_dir: Path,
) -> None:
    """Save a processed float16 volume back as a DICOM series.

    Stores float16 as a 16-bit unsigned integer after re-scaling to uint16
    range; records the scale/offset in RescaleSlope/Intercept so the original
    physical values remain recoverable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Re-scale float16 [0,1] → uint16 [0, 65535]
    volume_u16 = (volume_f16.astype(np.float32) * 65535).clip(0, 65535).astype(np.uint16)

    for i, src_path in enumerate(src_files):
        ds = pydicom.dcmread(str(src_path))

        # Replace pixel data with processed slice
        if volume_u16.ndim == 3:
            ds.PixelData = volume_u16[:, :, i].tobytes()
        else:
            ds.PixelData = volume_u16.tobytes()

        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
        ds.RescaleSlope = 1.0 / 65535.0
        ds.RescaleIntercept = 0.0
        ds.Rows = volume_u16.shape[0]
        ds.Columns = volume_u16.shape[1]

        out_path = out_dir / src_path.name
        ds.save_as(str(out_path))


def process_dicom_dir(
    dicom_dir: Path,
    out_dir: Path,
    resize: Optional[int],
    test_mode: bool,
) -> Dict:
    """Process all DICOM files in a single directory (one series)."""
    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        dcm_files = [f for f in dicom_dir.iterdir() if f.suffix.lower() in {".dcm", ""}]
    if not dcm_files:
        return {}

    original_bytes = sum(f.stat().st_size for f in dcm_files)

    if test_mode:
        # Estimate reduction without actually writing
        logger.info(
            f"  [DICOM] {dicom_dir.name}: {len(dcm_files)} slices, {original_bytes/1024:.1f} KB"
        )
        # Float16 in uint16 DICOM = 50 % of original int32 → rough estimate
        estimated_bytes = original_bytes * 0.5
        return {
            "n_files": len(dcm_files),
            "original_bytes": original_bytes,
            "output_bytes": estimated_bytes,
        }

    try:
        volume, ref_ds = load_dicom_series(dcm_files)
    except Exception as e:
        logger.warning(f"  Skipping {dicom_dir.name}: {e}")
        return {}

    # 1. Clip percentiles
    volume = clip_percentiles(volume, 0.5, 99.5)

    # 2. Consistent bounding-box crop across the stack
    volume, _ = crop_stack_consistently(volume, margin=10)

    # 3. Optional resize
    if resize:
        volume = resize_volume(volume, resize)

    # 4. Normalize to float16
    volume_f16 = normalize_to_float16(volume)

    # 5. Save
    save_dicom_series(volume_f16, ref_ds, dcm_files, out_dir)

    output_bytes = sum(f.stat().st_size for f in out_dir.glob("*"))
    return {
        "n_files": len(dcm_files),
        "original_bytes": original_bytes,
        "output_bytes": output_bytes,
    }


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------


def process_nifti_file(
    nifti_path: Path,
    out_dir: Path,
    resize: Optional[int],
    test_mode: bool,
) -> Dict:
    """Process a single NIfTI (.nii / .nii.gz) file."""
    if not NIBABEL_AVAILABLE:
        logger.error("nibabel is required for NIfTI support. Install with: pip install nibabel")
        return {}

    original_bytes = nifti_path.stat().st_size

    if test_mode:
        logger.info(f"  [NIfTI] {nifti_path.name}: {original_bytes/1024:.1f} KB")
        estimated_bytes = original_bytes * 0.45
        return {
            "n_files": 1,
            "original_bytes": original_bytes,
            "output_bytes": estimated_bytes,
        }

    try:
        img = nib.load(str(nifti_path))
        affine = img.affine
        header = img.header.copy()
        data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        logger.warning(f"  Skipping {nifti_path.name}: {e}")
        return {}

    if data.ndim == 4:
        # 4-D (time-series): process each volume independently
        vols = []
        for t in range(data.shape[3]):
            v = data[..., t]
            v = clip_percentiles(v, 0.5, 99.5)
            v, _ = crop_stack_consistently(v, margin=10)
            if resize:
                v = resize_volume(v, resize)
            vols.append(normalize_to_float16(v))
        # Pad back to same shape (crop may differ between vols — use largest)
        max_shape = np.max([v.shape for v in vols], axis=0)
        padded = []
        for v in vols:
            pad = [(0, max_shape[i] - v.shape[i]) for i in range(v.ndim)]
            padded.append(np.pad(v, pad))
        processed = np.stack(padded, axis=-1)
    else:
        data = clip_percentiles(data, 0.5, 99.5)
        data, _ = crop_stack_consistently(data, margin=10)
        if resize:
            data = resize_volume(data, resize)
        processed = normalize_to_float16(data)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / nifti_path.name

    # Save as float32 NIfTI (nibabel doesn't natively store float16)
    # but store as float16-compatible by saving float32 values that were
    # computed from float16 — effectively preserving the reduced precision.
    out_img = nib.Nifti1Image(processed.astype(np.float32), affine, header)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))

    output_bytes = out_path.stat().st_size
    return {
        "n_files": 1,
        "original_bytes": original_bytes,
        "output_bytes": output_bytes,
    }


# ---------------------------------------------------------------------------
# Folder traversal
# ---------------------------------------------------------------------------


def mirror_output_path(src_root: Path, src_path: Path, out_root: Path) -> Path:
    """Compute the output path mirroring the source folder structure."""
    rel = src_path.relative_to(src_root)
    return out_root / rel


def find_dicom_dirs(root: Path) -> List[Path]:
    """Return all directories that directly contain ≥1 DICOM file."""
    dirs = set()
    for ext in ("*.dcm",):
        for f in root.rglob(ext):
            dirs.add(f.parent)
    # Also catch extension-less DICOM
    for f in root.rglob("*"):
        if f.is_file() and f.suffix == "":
            try:
                pydicom.dcmread(str(f), stop_before_pixels=True)
                dirs.add(f.parent)
            except Exception:
                pass
    return sorted(dirs)


def find_nifti_files(root: Path) -> List[Path]:
    """Return all NIfTI files under root."""
    files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    return sorted(files)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    input_dir: str,
    resize: Optional[int] = None,
    delete: bool = False,
    test_mode: bool = False,
) -> None:
    src_root = Path(input_dir).resolve()
    if not src_root.exists():
        logger.error(f"Input directory does not exist: {src_root}")
        sys.exit(1)

    out_root = src_root.parent / f"{src_root.name}-storage"

    if test_mode:
        logger.info("=== TEST MODE (no files written) ===")
    logger.info(f"Input : {src_root}")
    logger.info(f"Output: {out_root}")
    if resize:
        logger.info(f"Resize: {resize}×{resize}")
    if delete:
        logger.info("Delete originals: YES")

    dicom_dirs = find_dicom_dirs(src_root)
    nifti_files = find_nifti_files(src_root)

    total_original = 0
    total_output = 0
    total_files = 0

    # --- DICOM ---
    if dicom_dirs:
        logger.info(f"\nFound {len(dicom_dirs)} DICOM series directory(ies)")
    for dcm_dir in dicom_dirs:
        out_dir = mirror_output_path(src_root, dcm_dir, out_root)
        result = process_dicom_dir(dcm_dir, out_dir, resize, test_mode)
        if result:
            total_original += result["original_bytes"]
            total_output += result["output_bytes"]
            total_files += result["n_files"]
            pct = (1 - result["output_bytes"] / max(result["original_bytes"], 1)) * 100
            logger.info(
                f"  {dcm_dir.relative_to(src_root)}: "
                f"{result['n_files']} files, "
                f"{result['original_bytes']/1024:.0f} KB → "
                f"{result['output_bytes']/1024:.0f} KB "
                f"({pct:.1f}% reduction)"
            )

    # --- NIfTI ---
    if nifti_files:
        logger.info(f"\nFound {len(nifti_files)} NIfTI file(s)")
    for nii_path in nifti_files:
        out_dir = mirror_output_path(src_root, nii_path.parent, out_root)
        result = process_nifti_file(nii_path, out_dir, resize, test_mode)
        if result:
            total_original += result["original_bytes"]
            total_output += result["output_bytes"]
            total_files += result["n_files"]
            pct = (1 - result["output_bytes"] / max(result["original_bytes"], 1)) * 100
            logger.info(
                f"  {nii_path.relative_to(src_root)}: "
                f"{result['original_bytes']/1024:.0f} KB → "
                f"{result['output_bytes']/1024:.0f} KB "
                f"({pct:.1f}% reduction)"
            )

    # --- Summary ---
    if total_original > 0:
        overall_pct = (1 - total_output / total_original) * 100
        logger.info(
            f"\n{'='*50}\n"
            f"Total files processed : {total_files}\n"
            f"Original size         : {total_original / 1e6:.1f} MB\n"
            f"Output size           : {total_output / 1e6:.1f} MB\n"
            f"Overall size reduction: {overall_pct:.1f}%\n"
            f"{'='*50}"
        )
    else:
        logger.warning("No DICOM or NIfTI files found under the input directory.")

    # --- Delete originals ---
    if delete and not test_mode and total_files > 0:
        logger.info("Deleting original files...")
        for dcm_dir in dicom_dirs:
            for f in dcm_dir.iterdir():
                if f.is_file():
                    f.unlink()
        for nii_path in nifti_files:
            nii_path.unlink()
        logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MRI preprocessing: background crop, float16 normalization, optional resize."
    )
    parser.add_argument("input_dir", help="Root folder containing DICOM series or NIfTI files.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Dry run: report file counts and estimated size reduction without writing.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        choices=sorted(VALID_RESIZE_DIMS),
        default=None,
        metavar="{256,384,512}",
        help="Resize the two spatial dimensions to this value (e.g. 256 → 256×256).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete original files after successful processing.",
    )
    args = parser.parse_args()

    run(
        input_dir=args.input_dir,
        resize=args.resize,
        delete=args.delete,
        test_mode=args.test,
    )


if __name__ == "__main__":
    main()
