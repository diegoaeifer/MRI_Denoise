# Losses Directory

This module provides the objective functions used to optimize the neural networks during training.

## Files & Capabilities

*   **`composite.py`**: Contains the `CompositeLoss` class.
    *   **Capabilities**: Combines multiple complementary loss functions to achieve balanced denoising. Typically, it aggregates Charbonnier Loss (for robust L1-like penalties), SSIM (Structural Similarity Index), PSNR, and standard L1 loss.
    *   **How to Use**: Imported and instantiated in `train.py`. It takes the network's prediction and the clean target image, returning a combined scalar loss to call `.backward()` on.
*   **`auxiliary.py`**: Contains auxiliary loss definitions (if active).
    *   **Capabilities**: Handles additional loss constraints or secondary objectives that might be used alongside the primary composite loss.
    *   **How to Use**: Integrated into the training pipeline or `composite.py` when specific architectural features demand it.

## Antigravity Model Interaction
**READ-ONLY**. The Antigravity test suite will instantiate `CompositeLoss` for isolated forward/backward pass testing and loss convergence validation.
