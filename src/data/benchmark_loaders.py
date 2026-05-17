"""
Dataset loaders for the comprehensive MRI denoising benchmark.

Each loader yields one volume per sequence type (modality), normalized to [0, 1].
Volume shape convention: (H, W, D) float32.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

_IXI_ROOT = Path(r"C:\projetos\Datasets\IXI\all")
_LUMBAR_ROOT = Path(
    r"C:\projetos\Datasets\Lumbar_spine"
    r"\rsna-2024-lumbar-spine-degenerative-classification"
)
_MR4RAW_ROOT = Path(r"C:\projetos\mr4raw\data")


def _normalize(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = float(vol.min()), float(vol.max())
    denom = vmax - vmin
    if denom < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / denom).astype(np.float32)


class IXILoader:
    """Yields 1 volume per modality from IXI/all/*.nii.gz.

    File naming: IXI{id}-{site}-{scan_id}-{modality}.nii.gz
    Modalities present: T1, T2, PD, MRA
    """

    def __init__(self, root: Path | str = _IXI_ROOT):
        self.root = Path(root)

    def volumes(self) -> Iterator[dict]:
        import nibabel as nib

        modality_files: dict[str, list[Path]] = {}
        for f in sorted(self.root.glob("*.nii.gz")):
            stem = f.name.replace(".nii.gz", "")
            parts = stem.split("-")
            if len(parts) < 4:
                continue
            modality = parts[-1]
            modality_files.setdefault(modality, []).append(f)

        for modality in sorted(modality_files):
            first_file = modality_files[modality][0]
            try:
                img = nib.load(str(first_file))
                vol = img.get_fdata(dtype=np.float32)
                subject_id = first_file.name.split("-")[0]
                yield {
                    "volume": _normalize(vol),
                    "dataset": "IXI",
                    "modality": modality,
                    "subject_id": subject_id,
                }
            except Exception as e:
                log.warning(f"IXILoader: skipping {first_file.name}: {e}")


class LumbarLoader:
    """Yields 1 volume per series_description from the RSNA Lumbar Spine dataset.

    Reads train_series_descriptions.csv, loads DICOMs from
    train_images/{study_id}/{series_id}/*.dcm, stacks into (H, W, D).
    """

    def __init__(self, root: Path | str = _LUMBAR_ROOT):
        self.root = Path(root)

    def volumes(self) -> Iterator[dict]:
        import pandas as pd
        import pydicom

        desc_csv = self.root / "train_series_descriptions.csv"
        if not desc_csv.exists():
            log.warning(f"LumbarLoader: {desc_csv} not found")
            return

        df = pd.read_csv(desc_csv, dtype=str)

        # Only keep rows where train_images/{study_id} actually exists on disk
        images_root = self.root / "train_images"
        df = df[df["study_id"].apply(
            lambda sid: (images_root / str(sid)).is_dir()
        )]

        for desc, group in sorted(df.groupby("series_description")):
            first_row = group.sort_values("study_id").iloc[0]
            study_id = str(first_row["study_id"])
            series_id = str(first_row["series_id"])
            series_dir = self.root / "train_images" / study_id / series_id

            dcm_files = sorted(series_dir.glob("*.dcm"))
            if not dcm_files:
                log.warning(f"LumbarLoader: no DICOMs in {series_dir}")
                continue

            slices = []
            for f in dcm_files:
                try:
                    ds = pydicom.dcmread(str(f))
                    slices.append(ds.pixel_array.astype(np.float32))
                except Exception as e:
                    log.warning(f"LumbarLoader: skipping {f.name}: {e}")

            if not slices:
                continue

            vol = np.stack(slices, axis=-1)  # (H, W, D)
            yield {
                "volume": _normalize(vol),
                "dataset": "Lumbar",
                "modality": str(desc),
                "subject_id": study_id,
            }


class MR4RawLoader:
    """Yields 1 volume per modality from mr4raw/data/*.h5.

    File naming: {date}_{modality_tag}.h5  e.g. 2022061001_T101.h5
    Modality extraction: FLAIR* -> FLAIR, T1* -> T1, T2* -> T2
    Reads reconstruction_rss (D, H, W) -> transpose to (H, W, D).
    """

    def __init__(self, root: Path | str = _MR4RAW_ROOT):
        self.root = Path(root)

    def volumes(self) -> Iterator[dict]:
        import h5py

        modality_files: dict[str, list[Path]] = {}
        for f in sorted(self.root.glob("*.h5")):
            stem = f.stem
            tag = stem.split("_", 1)[1] if "_" in stem else stem
            if tag.upper().startswith("FLAIR"):
                modality = "FLAIR"
            elif tag.upper().startswith("T1"):
                modality = "T1"
            elif tag.upper().startswith("T2"):
                modality = "T2"
            else:
                modality = tag
            modality_files.setdefault(modality, []).append(f)

        for modality in sorted(modality_files):
            first_file = modality_files[modality][0]
            try:
                with h5py.File(str(first_file), "r") as hf:
                    vol_dhw = hf["reconstruction_rss"][:].astype(np.float32)  # (D, H, W)
                vol = np.transpose(vol_dhw, (1, 2, 0))  # (H, W, D)
                subject_id = first_file.stem.split("_")[0]
                yield {
                    "volume": _normalize(vol),
                    "dataset": "MR4Raw",
                    "modality": modality,
                    "subject_id": subject_id,
                }
            except Exception as e:
                log.warning(f"MR4RawLoader: skipping {first_file.name}: {e}")
