"""
MONAI-style datalist builder for DICOM and NIfTI datasets.

Wraps the patient-wise split logic from DICOMLoader (irreplaceable custom logic)
and emits MONAI-compatible datalists: [{"image": path}, ...].

MONAI downstream uses LoadImaged with PydicomReader/NibabelReader (auto-selected).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def build_datalist(data_config: dict) -> Dict[str, List[Dict[str, str]]]:
    """
    Scan data directory and emit a MONAI-style datalist.

    Args:
        data_config: dict with keys:
            - raw_path: str, root directory containing DICOMs or NIfTI files
            - split_ratios: dict (train/val/test), default 0.8/0.1/0.1
            - seed: int, default 42
            - limit: int or None
            - cache: bool, default True

    Returns:
        {"train": [{"image": path}, ...], "val": [...], "test": [...]}
    """
    raw_path = Path(data_config["raw_path"])

    # Detect mode: DICOM or NIfTI
    nifti_files = list(raw_path.rglob("*.nii")) + list(raw_path.rglob("*.nii.gz"))
    dicom_files = list(raw_path.rglob("*.dcm")) + list(raw_path.rglob("*.DCM"))

    if nifti_files and not dicom_files:
        logger.info(f"Detected NIfTI dataset: {len(nifti_files)} files")
        splits = _split_nifti(nifti_files, data_config)
    elif dicom_files or not nifti_files:
        logger.info(f"Detected DICOM dataset (or empty, will scan for extensionless)")
        splits = _split_dicom(raw_path, data_config)
    else:
        # Mixed — prefer NIfTI as primary
        logger.info("Mixed dataset — treating NIfTI as primary")
        splits = _split_nifti(nifti_files, data_config)

    # Convert to MONAI datalist format
    return {
        split: [{"image": str(p)} for p in paths]
        for split, paths in splits.items()
    }


def _split_nifti(
    files: List[Path], data_config: dict
) -> Dict[str, List[Path]]:
    """Patient-agnostic split for NIfTI files (no PatientID in NIfTI)."""
    import random
    ratios = data_config.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = data_config.get("seed", 42)
    limit = data_config.get("limit", None)

    random.seed(seed)
    files = list(files)
    random.shuffle(files)

    if limit:
        files = files[:limit]

    n = len(files)
    n_train = int(n * ratios.get("train", 0.8))
    n_val = int(n * ratios.get("val", 0.1))

    return {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }


def _split_dicom(
    data_path: Path, data_config: dict
) -> Dict[str, List[Path]]:
    """
    Patient-wise DICOM split. Keeps custom DICOMLoader logic.
    Returns dict of path lists (individual DICOM files).
    """
    # Reuse existing DICOMLoader scan logic
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))  # ensure src is importable
    from src.data.loader import DICOMLoader

    loader = DICOMLoader(
        data_path=str(data_path),
        seed=data_config.get("seed", 42),
        split_ratios=data_config.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1}),
        limit=data_config.get("limit", None),
        cache=data_config.get("cache", True),
    )
    splits = loader.create_splits()
    return {k: [Path(p) for p in v] for k, v in splits.items()}


def save_datalist(datalist: Dict[str, List[Dict]], output_dir: str) -> None:
    """Save datalist to JSON for reproducibility and MONAI bundle compatibility."""
    os.makedirs(output_dir, exist_ok=True)
    for split, items in datalist.items():
        path = os.path.join(output_dir, f"{split}.json")
        with open(path, "w") as f:
            json.dump(items, f, indent=2)
        logger.info(f"Saved {split} datalist ({len(items)} items) → {path}")


def load_datalist(datalist_dir: str) -> Dict[str, List[Dict]]:
    """Load pre-built datalist JSON files (for reproducible runs)."""
    result = {}
    for split in ("train", "val", "test"):
        path = os.path.join(datalist_dir, f"{split}.json")
        if os.path.exists(path):
            with open(path) as f:
                result[split] = json.load(f)
    return result
