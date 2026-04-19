import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch
import pytest

# Use patch.dict for sys.modules to avoid poisoning other tests
# We define a fixture to patch modules before importing our pipeline
@pytest.fixture(autouse=True, scope="module")
def mock_dependencies():
    mock_modules = {
        'torch': MagicMock(),
        'pydicom': MagicMock(),
        'PIL': MagicMock(),
        'scipy': MagicMock(),
        'scipy.ndimage': MagicMock(),
        'models': MagicMock(),
        'models.factory': MagicMock()
    }

    with patch.dict('sys.modules', mock_modules):
        yield

# Import pipeline inside the tests or after the fixture is set up
# We need it at module level, so we use the dict patch in the fixture,
# but for the actual import we do it inside the function or dynamically
# in another fixture.

@pytest.fixture
def mock_pipeline(mock_dependencies):
    if 'src' not in sys.path:
        sys.path.insert(0, os.path.abspath('src'))

    from pipeline import DenoisePipeline

    config = {'models': {}}
    with patch('pipeline.get_model'):
        with patch('pipeline.torch.device', return_value='cpu'):
            pipeline = DenoisePipeline(model_name='dummy_model', config=config, device='cpu')

            # Mock denoise_image to return the image slightly scaled, so we can track the normalization logic
            # Normalization brings the array to [0, 1] range.
            def mock_denoise(x, sigma=0.05):
                return x * 0.9

            pipeline.denoise_image = MagicMock(side_effect=mock_denoise)
            return pipeline

def test_process_dicom(mock_pipeline):
    import pydicom

    mock_ds = MagicMock()

    # Create realistic test data where quantiles will be calculated properly
    # Using float32 for initial raw calculation mimicking pydicom's output range 0-4095
    base_data = np.random.randint(100, 2000, size=(256, 256)).astype(np.uint16)

    # ensure p1 and p99 are known
    base_data[0:10, 0:10] = 100 # min
    base_data[246:256, 246:256] = 1900 # max

    mock_ds.pixel_array = base_data
    pydicom.dcmread.return_value = mock_ds

    result = mock_pipeline.process_dicom('dummy_path.dcm', output_path='dummy_output.dcm')

    pydicom.dcmread.assert_called_once_with('dummy_path.dcm')
    mock_pipeline.denoise_image.assert_called_once()
    mock_ds.save_as.assert_called_once_with('dummy_output.dcm')

    # Assert return type matches expected type in process_dicom
    assert result.dtype == np.uint16
    assert result.shape == (256, 256)

    # Because our denoise mock multiplies by 0.9, values should have changed but remain in valid 0-65535 uint16 range
    # e.g. the maximum scaled value should be around 0.9 * max_val roughly
    assert np.max(result) <= 2000

def test_process_dicom_with_noise_estimation(mock_pipeline):
    import pydicom

    mock_ds = MagicMock()
    base_data = np.random.randint(100, 2000, size=(100, 100)).astype(np.uint16)
    mock_ds.pixel_array = base_data
    pydicom.dcmread.return_value = mock_ds

    mock_pipeline.estimate_noise_mad = MagicMock(return_value=0.1)

    mock_pipeline.process_dicom('dummy_path.dcm', estimate_noise='mad')

    mock_pipeline.estimate_noise_mad.assert_called_once()
    mock_pipeline.denoise_image.assert_called_once()
    call_args = mock_pipeline.denoise_image.call_args
    assert call_args[1]['sigma'] == 0.1
