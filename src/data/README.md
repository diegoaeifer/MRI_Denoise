# Data Directory

This module is responsible for handling raw DICOM MRI images, performing data cleaning, applying augmentations, and feeding tensors into the neural networks.

## Files & Capabilities

*   **`dataset.py`**: Contains the `MRI_DICOM_Dataset` class and the `collate_fn`.
    *   **Capabilities**: Loads images efficiently, applies 16-bit percentile-based normalization (clipping extreme values for dynamic range optimization), injects spatially varying synthetic noise (generating the "sigma map"), and manages dataset splits.
    *   **How to Use**: Imported and instantiated in `train.py` and `inference.py`. It takes a list of file paths and a configuration dictionary.
*   **`loader.py`**: Contains the `DICOMLoader` utility class.
    *   **Capabilities**: A low-level utility to safely read `pydicom` pixel arrays and handle specific DICOM metadata quirks.
    *   **How to Use**: Used internally by `dataset.py` to extract raw NumPy arrays from `.dcm` files.
*   **`transforms.py`**: Defines MRI-specific data augmentations.
    *   **Capabilities**: Performs interpolation-free ±90° discrete rotations to prevent aliasing, vertical/horizontal flipping, and anisotropy simulations.
    *   **How to Use**: Functions are called within `dataset.py` during the `__getitem__` pipeline based on probabilities defined in `config_data.yaml`.
*   **`check_background.py`**: A specialized script for massive data cleaning.
    *   **Capabilities**: Rapidly scans through directories of DICOM files to identify and optionally delete "flat" or mostly black uninformative images that hinder model convergence. It uses multiprocessing for high throughput.
    *   **How to Use**: Run as a standalone script from the command line:
        ```bash
        python src/data/check_background.py --data_path data/IXI --delete --batch_size 1000
        ```
