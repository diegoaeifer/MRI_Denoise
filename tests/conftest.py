import os
import tempfile
import pytest
import numpy as np

try:
    import torch
    from torch import nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False


@pytest.fixture
def torch_device():
    """Auto-detect GPU or use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_tensor_2channel():
    """Create a dummy 2-channel tensor (noisy_image + sigma_map)."""
    batch_size, height, width = 2, 256, 256
    image = torch.randn(batch_size, 1, height, width)
    sigma_map = torch.rand(batch_size, 1, height, width) * 0.1  # Values in [0, 0.1]
    return torch.cat([image, sigma_map], dim=1)


@pytest.fixture
def dummy_dicom_file():
    """Create a small dummy DICOM file for testing."""
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
    except ImportError:
        pytest.skip("pydicom not installed")

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        file_path = tmp.name

    ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.SeriesInstanceUID = "1.2.3.4"
    ds.Modality = "MR"
    ds.SamplesperPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 256
    ds.Columns = 256
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.random.randint(0, 4096, (256, 256), dtype=np.uint16).tobytes()

    ds.save_as(file_path, write_like_original=False)

    yield file_path

    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture
def dummy_nifti_file():
    """Create a small dummy NIfTI file for testing."""
    try:
        import nibabel as nib
    except ImportError:
        pytest.skip("nibabel not installed")

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        file_path = tmp.name

    data = np.random.rand(64, 64, 64).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, file_path)

    yield file_path

    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)


@pytest.fixture
def dummy_model():
    """Create a minimal 2-channel->1-channel model for testing."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    return DummyModel()


@pytest.fixture
def dummy_config():
    """Create a valid PipelineConfig for testing."""
    from src.config import get_defaults

    return get_defaults()


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_dataloader(dummy_tensor_2channel):
    """Create a mock dataloader that yields dummy batches."""

    def create_mock_loader(num_batches=3):
        for _ in range(num_batches):
            batch = {
                "input": dummy_tensor_2channel,
                "target": dummy_tensor_2channel[:, :1, :, :],  # Only image channel
                "path": "dummy_path.dcm",
            }
            yield batch

    return create_mock_loader
