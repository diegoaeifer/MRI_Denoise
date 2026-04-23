import os
import tempfile
import pytest
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import logging

from src.data.loader import DICOMLoader


@pytest.fixture
def temp_dicom_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_dummy_dicom(filepath, patient_id="123", series_uid="456"):
    # Create a minimal valid DICOM file
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
    file_meta.ImplementationClassUID = UID("1.2.3.4")

    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.SeriesInstanceUID = series_uid
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    ds.save_as(filepath)


def test_dicom_loader_valid_files(temp_dicom_dir):
    # Create some valid DICOM files
    create_dummy_dicom(os.path.join(temp_dicom_dir, "file1.dcm"), patient_id="P1", series_uid="S1")
    create_dummy_dicom(os.path.join(temp_dicom_dir, "file2.dcm"), patient_id="P1", series_uid="S1")
    create_dummy_dicom(os.path.join(temp_dicom_dir, "file3.dcm"), patient_id="P2", series_uid="S2")

    loader = DICOMLoader(data_path=temp_dicom_dir, cache=False)
    patient_registry, series_registry = loader.scan_directory()

    assert len(patient_registry) == 2
    assert "P1" in patient_registry and "P2" in patient_registry
    assert len(series_registry) == 2
    assert "S1" in series_registry and "S2" in series_registry
    assert len(series_registry["S1"]) == 2
    assert len(series_registry["S2"]) == 1


def test_dicom_loader_invalid_file(temp_dicom_dir, caplog):
    # Create a valid file
    create_dummy_dicom(os.path.join(temp_dicom_dir, "valid.dcm"), patient_id="P1", series_uid="S1")

    # Create an invalid file (not a DICOM file but ends with .dcm)
    invalid_filepath = os.path.join(temp_dicom_dir, "invalid.dcm")
    with open(invalid_filepath, "wb") as f:
        f.write(b"this is not a valid dicom file data")

    loader = DICOMLoader(data_path=temp_dicom_dir, cache=False)

    with caplog.at_level(logging.WARNING):
        patient_registry, series_registry = loader.scan_directory()

    # Verify that the valid file was loaded successfully
    assert len(patient_registry) == 1
    assert "P1" in patient_registry
    assert len(series_registry) == 1
    assert len(series_registry["S1"]) == 1

    # Verify that the invalid file triggered a warning message
    warning_messages = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]
    assert any(
        "Invalid DICOM file" in msg and "invalid.dcm" in msg for msg in warning_messages
    ), f"Expected warning for invalid DICOM file not found. Logs: {warning_messages}"
