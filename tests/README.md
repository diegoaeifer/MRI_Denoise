# Tests Directory

This directory contains the unit tests and testing utilities designed to ensure the reliability and correctness of the data pipelines and codebase logic. It uses the `pytest` framework.

## Files & Capabilities

*   **`gen_data.py`**: A utility script to generate synthetic or dummy DICOM data using `pydicom` for the purpose of testing without needing actual MRI datasets.
*   **`test_flipping.py`**: Tests the horizontal and vertical flipping data augmentations implemented in `src/data/transforms.py`.
*   **`test_loader.py`**: Verifies that `src/data/loader.py` correctly reads, parses, and formats DICOM pixel arrays.
*   **`test_model_factory.py`**: Ensures `src/models/factory.py` successfully initializes every supported model architecture (NAFNet, DRUNet, SCUNet, UNet) given the configurations without dimension errors.
*   **`test_paths.py`**: Checks that critical directory paths, configurations, and environment setups are correctly established.
*   **`test_rotation.py`**: Rigorously tests the discrete ±90° rotation augmentation to ensure it operates interpolation-free and without aliasing.
*   **`test_train_tqdm.py`**: Verifies that the training loop dependencies (like progress bars) initialize gracefully even in diverse environments.

## How to Use
Run the tests from the root of the project using `pytest`. Make sure to set the python path so the `src` directory is resolved correctly:

```bash
PYTHONPATH=. python3 -m pytest tests/
```
