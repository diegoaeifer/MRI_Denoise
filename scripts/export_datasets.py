"""Export MRI datasets from C:\\projetos\\Datasets to D:\\Dataset MRI.

Scans source datasets, copies DICOMs preserving study/series hierarchy,
records NIfTI/H5 paths in-place, and rebuilds manifest.csv.

Usage:
    python scripts/export_datasets.py [--dry_run] [--datasets lumbar abdomen ixi all]
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC_ROOT = Path(r"C:\projetos\Datasets")
DST_ROOT = Path(r"D:\Dataset MRI")
MANIFEST_PATH = DST_ROOT / "manifest.csv"

MANIFEST_COLS = [
    "file_path", "source_dataset", "subject_id", "anatomy",
    "sequence", "plane", "is_3d", "vendor", "field_strength_T",
    "slice_thickness_mm", "pixel_spacing_mm", "rows", "cols",
    "original_format", "has_noise_map", "pathology", "split",
]

DATASET_MAP = {
    "lumbar": {
        "src": SRC_ROOT / "Lumbar_spine" / "rsna-2024-lumbar-spine-degenerative-classification" / "train_images",
        "dst_anatomy": "spine",
        "vendor": "unknown",
        "field_T": "unknown",
        "anatomy": "spine",
        "source_dataset": "Lumbar_spine",
        "format": "dicom",
        "pathology": "degenerative",
        "max_files": 5000,
    },
    "abdomen": {
        "src": SRC_ROOT / "MRIabdomen" / "dicoms",
        "dst_anatomy": "body",
        "vendor": "siemens",
        "field_T": "unknown",
        "anatomy": "body",
        "source_dataset": "MRIabdomen",
        "format": "dicom",
        "pathology": "normal",
        "max_files": 5000,
    },
    "rsna_brain": {
        "src": SRC_ROOT / "rsna-miccai-brain-tumor-radiogenomic-classification",
        "dst_anatomy": "brain",
        "vendor": "unknown",
        "field_T": "unknown",
        "anatomy": "brain",
        "source_dataset": "rsna-miccai-brain-tumor-radiogenomic-classification",
        "format": "dicom",
        "pathology": "tumor",
        "max_files": 5000,
    },
    "ixi": {
        "src": SRC_ROOT / "IXI",
        "dst_anatomy": "brain",
        "vendor": "philips",
        "field_T": "1.5",
        "anatomy": "brain",
        "source_dataset": "IXI",
        "format": "nifti",
        "pathology": "normal",
        "max_files": 5000,
    },
    "totalseg": {
        "src": SRC_ROOT / "TotalsegmentatorMRI_dataset_v200",
        "dst_anatomy": "body",
        "vendor": "siemens",
        "field_T": "1.5",
        "anatomy": "body",
        "source_dataset": "TotalsegmentatorMRI",
        "format": "nifti",
        "pathology": "normal",
        "max_files": 3000,
    },
    "openmind": {
        "src": SRC_ROOT / "openmind",
        "dst_anatomy": "brain",
        "vendor": "unknown",
        "field_T": "unknown",
        "anatomy": "brain",
        "source_dataset": "openmind",
        "format": "dicom",
        "pathology": "normal",
        "max_files": 2000,
    },
    "shifts_ms": {
        "src": SRC_ROOT / "shifts_MS",
        "dst_anatomy": "brain",
        "vendor": "unknown",
        "field_T": "1.5",
        "anatomy": "brain",
        "source_dataset": "shifts_MS",
        "format": "nifti",
        "pathology": "ms",
        "max_files": 2000,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_files(src: Path, extensions: tuple, max_files: int) -> list[Path]:
    """Walk src, collecting files with given extensions up to max_files."""
    found: list[Path] = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(extensions):
                found.append(Path(root) / f)
                if len(found) >= max_files:
                    return found
    return found


def copy_file(src_path: Path, dst_path: Path, dry_run: bool) -> bool:
    """Copy src to dst, creating parent dirs. Returns True if file was copied."""
    if dst_path.exists():
        return False  # already there
    if dry_run:
        return True
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


def record_in_place(src_path: Path) -> bool:
    """For NIfTI/H5: record the source path directly (no copy needed)."""
    return src_path.exists()


def build_record(
    file_path: Path,
    cfg: dict,
    subject_id: str = "",
    is_3d: bool = False,
) -> dict:
    return {
        "file_path": str(file_path),
        "source_dataset": cfg["source_dataset"],
        "subject_id": subject_id,
        "anatomy": cfg["anatomy"],
        "sequence": "",
        "plane": "",
        "is_3d": str(is_3d),
        "vendor": cfg["vendor"],
        "field_strength_T": cfg["field_T"],
        "slice_thickness_mm": "",
        "pixel_spacing_mm": "",
        "rows": "",
        "cols": "",
        "original_format": cfg["format"],
        "has_noise_map": "False",
        "pathology": cfg["pathology"],
        "split": "unassigned",
    }


# ---------------------------------------------------------------------------
# Per-dataset export logic
# ---------------------------------------------------------------------------

def export_dicom_dataset(name: str, cfg: dict, dry_run: bool) -> list[dict]:
    src: Path = cfg["src"]
    if not src.exists():
        print(f"  [SKIP] {name}: source not found at {src}")
        return []

    dst_base = DST_ROOT / cfg["dst_anatomy"] / f"{cfg['vendor']}_{cfg['field_T']}T" / cfg["source_dataset"]
    files = collect_files(src, (".dcm",), cfg["max_files"])
    print(f"  {name}: {len(files)} DICOM files found, copying to {dst_base}")

    records = []
    copied = 0
    skipped = 0
    for i, f in enumerate(files, 1):
        # Build flat destination name: preserve series folder + original filename
        # to avoid collision, use relative path components joined with underscore
        rel = f.relative_to(src)
        parts = rel.parts
        # Use last 2 path components for uniqueness (series_id/filename)
        if len(parts) >= 2:
            dst_name = f"{parts[-2]}_{parts[-1]}"
        else:
            dst_name = parts[-1]
        dst_path = dst_base / dst_name

        subject_id = parts[0] if len(parts) >= 2 else "unknown"

        if copy_file(f, dst_path, dry_run):
            copied += 1
        else:
            skipped += 1

        records.append(build_record(dst_path, cfg, subject_id=subject_id))

        if i % 500 == 0:
            print(f"    ... {i}/{len(files)} ({copied} copied, {skipped} skipped)")

    print(f"  Done {name}: {copied} copied, {skipped} already existed")
    return records


def export_nifti_dataset(name: str, cfg: dict) -> list[dict]:
    """Record NIfTI/H5 paths in-place — no copy."""
    src: Path = cfg["src"]
    if not src.exists():
        print(f"  [SKIP] {name}: source not found at {src}")
        return []

    exts = (".nii", ".nii.gz", ".h5", ".hdf5")
    files = collect_files(src, exts, cfg["max_files"])
    print(f"  {name}: {len(files)} NIfTI/H5 files (recording paths in-place)")

    records = []
    for f in files:
        subject_id = f.parent.name
        records.append(build_record(f, cfg, subject_id=subject_id, is_3d=True))

    print(f"  Done {name}: {len(records)} records added")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_existing_manifest() -> set[str]:
    """Return set of file_path strings already in manifest."""
    if not MANIFEST_PATH.exists():
        return set()
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["file_path"] for row in reader}


def save_manifest(records: list[dict], append: bool = True):
    mode = "a" if (append and MANIFEST_PATH.exists()) else "w"
    with open(MANIFEST_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if mode == "w":
            writer.writeheader()
        writer.writerows(records)
    print(f"Manifest: {len(records)} records {'appended' if append else 'written'} -> {MANIFEST_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true", help="Count files, don't copy")
    p.add_argument(
        "--datasets", nargs="+",
        default=["lumbar", "abdomen", "ixi", "rsna_brain", "totalseg", "openmind", "shifts_ms"],
        help="Which datasets to process (or 'all')"
    )
    args = p.parse_args()

    if "all" in args.datasets:
        args.datasets = list(DATASET_MAP.keys())

    existing = load_existing_manifest()
    print(f"Existing manifest entries: {len(existing)}")
    if args.dry_run:
        print("[DRY RUN] No files will be copied.\n")

    all_new_records: list[dict] = []

    for name in args.datasets:
        if name not in DATASET_MAP:
            print(f"Unknown dataset: {name}")
            continue
        cfg = DATASET_MAP[name]
        print(f"\n--- {name} ---")

        if cfg["format"] == "dicom":
            records = export_dicom_dataset(name, cfg, dry_run=args.dry_run)
        else:
            records = export_nifti_dataset(name, cfg)

        # Deduplicate against existing manifest
        new_records = [r for r in records if r["file_path"] not in existing]
        all_new_records.extend(new_records)
        existing.update(r["file_path"] for r in new_records)
        print(f"  New manifest entries: {len(new_records)}")

    print(f"\nTotal new records to add: {len(all_new_records)}")

    if not args.dry_run and all_new_records:
        save_manifest(all_new_records, append=True)

    if args.dry_run:
        print("\n[DRY RUN] Summary complete. Re-run without --dry_run to execute.")


if __name__ == "__main__":
    main()
