# Utils Directory

This module contains helper functions and utilities to support training and evaluation.

## Files & Capabilities

*   **`metrics.py`**: Contains metric calculation functions such as `calculate_roi_snr`.
    *   **Capabilities**: Provides standardized mathematical evaluations of image quality, calculating Signal-to-Noise Ratio (SNR) over regions of interest, helping to quantify denoising performance beyond simple loss functions.
    *   **How to Use**: Imported and utilized within the evaluation loops inside `train.py` or during post-processing in inference scripts.
