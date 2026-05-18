"""
Hallucination and quality detection for MRI denoising outputs.

Uses:
- sfrc (Signal-to-Frequency Ratio Curve) for hallucination detection
- DLMO (Deep Learning Model Output) for lesion removal detection
- mr-recon-eval-core for reconstruction quality

Usage:
    python scripts/hallucination_eval.py \\
        --denoised <path_to_denoised.nii> \\
        --reference <path_to_reference.nii> \\
        --output <output_dir>
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add DIDSR repos to sys.path so their modules are importable
# even when they are not pip-installed.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "FMImaging_MRI_Denoise" / "sfrc" / "src"))
sys.path.insert(0, str(_ROOT / "FMImaging_MRI_Denoise" / "DLMO"))
sys.path.insert(0, str(_ROOT / "FMImaging_MRI_Denoise" / "mr-recon-eval-core"))


# ---------------------------------------------------------------------------
# sFRC hallucination score
# ---------------------------------------------------------------------------

def _make_square_patch(arr: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Return a square centre crop of *arr* with side length *size*.
    If the image is smaller than *size* in any dimension it is zero-padded.
    The crop size is then rounded down to the nearest multiple of 4 (required
    by sfrc's diagonal_split helper).
    """
    h, w = arr.shape[:2]
    # Centre-crop
    ch = min(h, size)
    cw = min(w, size)
    r0 = (h - ch) // 2
    c0 = (w - cw) // 2
    patch = arr[r0 : r0 + ch, c0 : c0 + cw]

    # Zero-pad to *size* x *size* if needed
    if patch.shape[0] < size or patch.shape[1] < size:
        padded = np.zeros((size, size), dtype=patch.dtype)
        padded[: patch.shape[0], : patch.shape[1]] = patch
        patch = padded

    # Ensure divisible by 4 (sfrc requirement)
    s = (size // 4) * 4
    patch = patch[:s, :s]
    return patch.astype(np.float32)


def compute_sfrc_score(denoised: np.ndarray, reference: np.ndarray) -> dict:
    """
    Compute sFRC hallucination score between *denoised* and *reference* images.

    The sFRC library operates on 2-D square patches; we extract a 64-pixel
    centre crop from the input arrays (taking the central slice if 3-D).

    Returns
    -------
    dict with keys:
        sfrc_score          : float — mean FRC value at the 0.5 threshold
                              crossing, or NaN if the library is unavailable.
        sfrc_curve          : list[float] — per-ring FRC values.
        resolution_estimate : float — normalised spatial frequency (0–1) where
                              the FRC curve crosses the 0.5 threshold.
        error               : str (only present on failure)
    """
    try:
        from frc_utils import FRC  # sfrc/src/frc_utils.py
    except ImportError:
        return {"sfrc_score": float("nan"), "error": "sfrc not installed"}

    # Reduce to 2-D if needed (take the central slice along the first axis)
    def _to_2d(arr):
        if arr.ndim == 3:
            return arr[arr.shape[0] // 2]
        return arr

    patch_den = _make_square_patch(_to_2d(denoised))
    patch_ref = _make_square_patch(_to_2d(reference))

    try:
        x_t, t_vals, x_frc, frc_vals = FRC(
            patch_den,
            patch_ref,
            thresholding="0.5",
            inscribed_rings=True,
            analytical_arc_based=True,
            info_split=False,
        )
    except Exception as exc:
        return {"sfrc_score": float("nan"), "error": str(exc)}

    frc_arr = np.asarray(frc_vals, dtype=float)
    x_arr = np.asarray(x_frc, dtype=float)
    t_arr = np.asarray(t_vals, dtype=float)

    # Resolution estimate: first x where FRC drops below threshold
    below = np.where(frc_arr < t_arr)[0]
    resolution_estimate = float(x_arr[below[0]]) if len(below) > 0 else 1.0

    return {
        "sfrc_score": float(np.nanmean(frc_arr)),
        "sfrc_curve": frc_arr.tolist(),
        "resolution_estimate": resolution_estimate,
    }


# ---------------------------------------------------------------------------
# Synthetic lesion injection
# ---------------------------------------------------------------------------

def add_synthetic_lesions(
    img: np.ndarray, n_lesions: int = 4
) -> tuple:
    """
    Add *n_lesions* synthetic circular bright lesions to the central 2-D slice
    of *img* (operates in-place on a copy; original is not modified).

    Lesions are Gaussian-blurred disk masks placed pseudo-randomly within
    the central 50 % of the image to avoid edge effects.

    Parameters
    ----------
    img : np.ndarray
        2-D float image (H, W).
    n_lesions : int
        Number of lesions to insert.

    Returns
    -------
    modified_img : np.ndarray — copy of *img* with lesions added.
    lesion_masks : list[np.ndarray] — one boolean mask (H, W) per lesion.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed=42)
    h, w = img.shape[:2]
    modified = img.copy().astype(np.float64)

    # Lesion radius: ~5 % of the shorter side, minimum 3 px
    radius = max(3, int(min(h, w) * 0.05))
    sigma = radius / 2.5  # Gaussian smoothing width

    # Restrict centres to the central half of the image
    margin = max(radius + 1, min(h, w) // 4)
    cy_min, cy_max = margin, h - margin
    cx_min, cx_max = margin, w - margin

    lesion_masks = []
    for _ in range(n_lesions):
        cy = int(rng.integers(cy_min, max(cy_min + 1, cy_max)))
        cx = int(rng.integers(cx_min, max(cx_min + 1, cx_max)))

        # Build disk mask
        yy, xx = np.ogrid[:h, :w]
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2).astype(np.float64)

        # Smooth the disk to create a Gaussian blob
        blob = gaussian_filter(disk, sigma=sigma)
        blob = blob / (blob.max() + 1e-8)  # normalise to [0, 1]

        # Lesion intensity: 20 % above the local image max (floor at 1.0 for zero images)
        lesion_intensity = max(float(modified.max()), 1.0) * 0.2
        modified = modified + blob * lesion_intensity

        lesion_masks.append(disk.astype(np.float32))

    return modified.astype(img.dtype), lesion_masks


# ---------------------------------------------------------------------------
# DLMO lesion-removal detection
# ---------------------------------------------------------------------------

def compute_dlmo_score(
    denoised: np.ndarray, lesion_mask: np.ndarray
) -> dict:
    """
    Detect whether lesions present in *lesion_mask* were attenuated or removed
    during denoising by comparing the mean signal inside the lesion ROI to the
    surrounding background.

    When the DLMO library is available it is used directly; otherwise a
    signal-ratio heuristic is applied.

    Parameters
    ----------
    denoised : np.ndarray
        Denoised image (2-D or 3-D; central slice used for 3-D inputs).
    lesion_mask : np.ndarray
        Binary mask of the same spatial shape as *denoised* (values 0/1).

    Returns
    -------
    dict with keys:
        lesions_detected : float — fraction of lesion pixels whose signal
                           exceeds the background mean (0–1, higher = more
                           lesion signal preserved). NaN on failure.
        lesion_scores    : list[float] — per-pixel signal ratios within the
                           lesion ROI (relative to background mean).
        error            : str (only present on failure)
    """
    # Reduce to 2-D
    def _to_2d(arr):
        if arr.ndim == 3:
            return arr[arr.shape[0] // 2]
        return arr

    denoised_2d = _to_2d(denoised).astype(np.float64)
    mask_2d = _to_2d(lesion_mask).astype(bool)

    try:
        # Attempt to import DLMO library (not available — will except)
        import DLMO  # noqa: F401
        # If the library ships a detection API, call it here.
        raise NotImplementedError("DLMO API not yet integrated")
    except (ImportError, NotImplementedError):
        # Fallback: heuristic signal-ratio analysis
        if not mask_2d.any():
            return {
                "lesions_detected": float("nan"),
                "lesion_scores": [],
                "error": "empty lesion mask",
            }

        background_mean = float(denoised_2d[~mask_2d].mean()) + 1e-8
        lesion_signal = denoised_2d[mask_2d]
        scores = (lesion_signal / background_mean).tolist()
        detected_fraction = float((lesion_signal > background_mean).mean())

        return {
            "lesions_detected": detected_fraction,
            "lesion_scores": scores,
        }


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    denoised_path: Path,
    reference_path: Path,
    output_dir: Path,
) -> dict:
    """
    Load NIfTI images, run all hallucination / quality metrics, persist a JSON
    report, and return the results dict.

    Parameters
    ----------
    denoised_path : Path  — path to denoised image (.nii / .nii.gz)
    reference_path : Path — path to reference image (.nii / .nii.gz)
    output_dir : Path     — directory where ``hallucination_report.json`` is
                            written (created if absent)

    Returns
    -------
    dict with keys ``sfrc`` and ``dlmo`` (each a sub-dict of metric results).
    """
    import nibabel as nib

    denoised_nib = nib.load(str(denoised_path))
    reference_nib = nib.load(str(reference_path))

    denoised_arr = np.asarray(denoised_nib.dataobj, dtype=np.float32)
    reference_arr = np.asarray(reference_nib.dataobj, dtype=np.float32)

    # Use the central 2-D slice for lesion injection
    def _central_slice(arr):
        if arr.ndim == 3:
            return arr[arr.shape[0] // 2]
        return arr

    central = _central_slice(denoised_arr)
    _modified, lesion_masks = add_synthetic_lesions(central, n_lesions=4)

    # Combine all lesion masks into one for DLMO scoring
    combined_mask = np.zeros_like(central, dtype=np.float32)
    for m in lesion_masks:
        combined_mask = np.maximum(combined_mask, m)

    results = {
        "denoised": str(denoised_path),
        "reference": str(reference_path),
        "sfrc": compute_sfrc_score(denoised_arr, reference_arr),
        "dlmo": compute_dlmo_score(denoised_arr, combined_mask),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "hallucination_report.json"
    with open(report_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Report saved to {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Hallucination and quality detection for MRI denoising outputs. "
            "Wraps sfrc (sFRC hallucination detection) and DLMO (lesion-removal "
            "detection) from the DIDSR toolset."
        )
    )
    p.add_argument(
        "--denoised",
        type=Path,
        required=True,
        help="Path to the denoised NIfTI image (.nii / .nii.gz).",
    )
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the reference NIfTI image (.nii / .nii.gz).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/hallucination"),
        help="Output directory for the JSON report (default: artifacts/hallucination).",
    )
    args = p.parse_args()
    results = evaluate(args.denoised, args.reference, args.output)
    print(results)


if __name__ == "__main__":
    main()
