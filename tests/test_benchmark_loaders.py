"""TDD tests for benchmark dataset loaders."""
import numpy as np
import pytest


def _import_loaders():
    import importlib.util
    from pathlib import Path

    loader_path = Path(__file__).parent.parent / "src" / "data" / "benchmark_loaders.py"
    spec = importlib.util.spec_from_file_location("benchmark_loaders", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IXILoader, mod.LumbarLoader, mod.MR4RawLoader


# ── IXILoader ────────────────────────────────────────────────────────────────

def test_ixi_loader_yields_one_volume_per_modality(tmp_path):
    import nibabel as nib
    IXILoader, _, _ = _import_loaders()

    for modality in ("T1", "T2"):
        for subject in ("IXI015-HH-1258", "IXI016-Guys-0697"):
            fname = tmp_path / f"{subject}-{modality}.nii.gz"
            arr = np.random.rand(8, 8, 4).astype(np.float32)
            nib.save(nib.Nifti1Image(arr, np.eye(4)), str(fname))

    loader = IXILoader(root=tmp_path)
    vols = list(loader.volumes())

    assert len(vols) == 2
    assert {v["modality"] for v in vols} == {"T1", "T2"}
    assert vols[0]["volume"].ndim == 3
    assert vols[0]["volume"].dtype == np.float32
    assert vols[0]["volume"].min() >= 0.0
    assert vols[0]["volume"].max() <= 1.0
    assert vols[0]["dataset"] == "IXI"


def test_ixi_loader_picks_first_alphabetically(tmp_path):
    import nibabel as nib
    IXILoader, _, _ = _import_loaders()

    for subject in ("IXI100-HH-0001", "IXI015-HH-0002"):
        arr = np.ones((4, 4, 2), dtype=np.float32)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(tmp_path / f"{subject}-T1.nii.gz"))

    loader = IXILoader(root=tmp_path)
    vols = list(loader.volumes())
    assert len(vols) == 1
    assert vols[0]["subject_id"] == "IXI015"


def test_ixi_loader_empty_dir_yields_nothing(tmp_path):
    IXILoader, _, _ = _import_loaders()
    loader = IXILoader(root=tmp_path)
    assert list(loader.volumes()) == []


# ── LumbarLoader ─────────────────────────────────────────────────────────────

def _make_dcm(path, rows=8, cols=8):
    """Write a minimal valid DICOM file."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    from pydicom.dataset import FileDataset
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    pixel_data = np.random.randint(0, 1000, (rows, cols), dtype=np.uint16)
    ds.PixelData = pixel_data.tobytes()
    ds.save_as(str(path), write_like_original=False)


def test_lumbar_loader_yields_one_volume_per_modality(tmp_path):
    import pandas as pd
    _, LumbarLoader, _ = _import_loaders()

    rows_data = [
        {"study_id": "1111111111", "series_id": "100000001", "series_description": "Sagittal T2/STIR"},
        {"study_id": "2222222222", "series_id": "200000001", "series_description": "Sagittal T1"},
    ]
    for row in rows_data:
        series_dir = tmp_path / "train_images" / str(row["study_id"]) / str(row["series_id"])
        series_dir.mkdir(parents=True)
        for k in range(3):
            _make_dcm(series_dir / f"{k:04d}.dcm")

    pd.DataFrame(rows_data).to_csv(tmp_path / "train_series_descriptions.csv", index=False)

    loader = LumbarLoader(root=tmp_path)
    vols = list(loader.volumes())

    assert len(vols) == 2
    modalities = {v["modality"] for v in vols}
    assert modalities == {"Sagittal T2/STIR", "Sagittal T1"}
    assert vols[0]["volume"].ndim == 3
    assert vols[0]["volume"].dtype == np.float32
    assert vols[0]["volume"].min() >= 0.0
    assert vols[0]["volume"].max() <= 1.0
    assert vols[0]["dataset"] == "Lumbar"


def test_lumbar_loader_picks_first_study_alphabetically(tmp_path):
    import pandas as pd
    _, LumbarLoader, _ = _import_loaders()

    rows_data = [
        {"study_id": "9999999999", "series_id": "900000001", "series_description": "Axial T2"},
        {"study_id": "1000000000", "series_id": "100000001", "series_description": "Axial T2"},
    ]
    for row in rows_data:
        series_dir = tmp_path / "train_images" / str(row["study_id"]) / str(row["series_id"])
        series_dir.mkdir(parents=True)
        _make_dcm(series_dir / "0000.dcm")

    pd.DataFrame(rows_data).to_csv(tmp_path / "train_series_descriptions.csv", index=False)

    loader = LumbarLoader(root=tmp_path)
    vols = list(loader.volumes())
    assert len(vols) == 1
    assert vols[0]["subject_id"] == "1000000000"


# ── MR4RawLoader ─────────────────────────────────────────────────────────────

def test_mr4raw_loader_yields_one_volume_per_modality(tmp_path):
    import h5py
    _, _, MR4RawLoader = _import_loaders()

    for tag in ("T101", "T102", "T201", "FLAIR01"):
        fpath = tmp_path / f"2022061001_{tag}.h5"
        with h5py.File(str(fpath), "w") as hf:
            hf.create_dataset(
                "reconstruction_rss",
                data=np.random.rand(4, 8, 8).astype(np.float32),
            )

    loader = MR4RawLoader(root=tmp_path)
    vols = list(loader.volumes())

    assert len(vols) == 3
    assert {v["modality"] for v in vols} == {"T1", "T2", "FLAIR"}
    assert vols[0]["volume"].ndim == 3
    assert vols[0]["volume"].dtype == np.float32
    assert vols[0]["volume"].min() >= 0.0
    assert vols[0]["volume"].max() <= 1.0
    assert vols[0]["dataset"] == "MR4Raw"


def test_mr4raw_loader_volume_shape_is_hwD(tmp_path):
    """volume shape (H, W, D) — slices along last axis."""
    import h5py
    _, _, MR4RawLoader = _import_loaders()

    fpath = tmp_path / "2022061001_T101.h5"
    with h5py.File(str(fpath), "w") as hf:
        hf.create_dataset(
            "reconstruction_rss",
            data=np.ones((6, 16, 16), dtype=np.float32),  # (D=6, H=16, W=16)
        )

    loader = MR4RawLoader(root=tmp_path)
    vols = list(loader.volumes())
    assert vols[0]["volume"].shape == (16, 16, 6)  # (H, W, D)
