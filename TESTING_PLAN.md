# MRI Denoising Pipeline - Comprehensive Testing Implementation Plan

This document outlines the detailed plan to construct a suite of verification scripts that guarantee the correctness of the primary components within the MRI Denoising Pipeline: Augmentations, Losses, Inference, and Training.

We are delegating this implementation to the engineering team. Follow the phases outlined below.

---

## Phase 1: Existing Codebase Bugs (COMPLETED)

Before the verification scripts could be built, critical infrastructure issues were fixed to unblock the testing pipeline:

1.  **Synthetic DICOM Generation Issue**: The script `tests/gen_data.py` was generating test DICOMs without correctly specifying `TransferSyntaxUID`, `is_little_endian`, and `is_implicit_VR`. This caused `train.py --test` to crash with the error `No pixel_array found in DICOM`. **Fixed.**
2.  **`test_train_tqdm.py` failure**: The fallback import logic in `src/trainer.py` had a bug, causing test failures when testing environments lacking `tqdm`. **Fixed.**

---

## Phase 2: Losses Verification Script

**Goal**: Verify that our custom `CompositeLoss` module computes accurate values and performs efficiently compared to existing standards.

**Implementation Steps**:
1.  **Create Script**: Create a new file `tests/verify_losses.py`.
2.  **Sample Data**: Write a short function to download 2 standard grayscale images (e.g., standard Lena/Cameraman or simple dummy tensors). Normalize them to `[0, 1]`.
3.  **Noise Addition**: Take the first image and create a slightly noisy version of it (this will be the "prediction").
4.  **Metric Computation & Verification**:
    *   Initialize our `CompositeLoss` from `src.losses.composite`.
    *   Compute all supported sub-losses (L1, SSIM, MS-SSIM, PSNR, HaarPSI, LPIPS, DISTS, VGG) using our module.
    *   Instantiate native/reference implementations from the `piq` and `monai` packages directly.
    *   Assert that the metrics computed by our `CompositeLoss` match the direct outputs from `piq`/`monai` (within a small `float32` tolerance).
5.  **Runtime Profiling**: Add a simple timing loop using `time.time()` or `timeit` to ensure our wrapped losses do not add excessive overhead compared to calling `piq` directly.

---

## Phase 3: Augmentation Verification Script

**Goal**: Verify that spatial augmentations (e.g., rotation, flipping) correctly manipulate the image and that our metrics appropriately capture the loss in spatial correlation.

**Implementation Steps**:
1.  **Create Script**: Create a new file `tests/verify_augmentations.py`.
2.  **Sample Data**: Generate or load 10 clean DICOM `pixel_array`s.
3.  **Applying Augmentations**:
    *   For each image, apply horizontal flip, vertical flip, and 90-degree rotations using functions from `src.data.transforms`.
4.  **Metric Impact Verification**:
    *   Compute the Structural Similarity metrics (SSIM, MS-SSIM, PSNR, HaarPSI, LPIPS, DISTS) comparing the **original clean image** against the **augmented image**.
    *   Assert that the Structural/Perceptual similarity metrics **drop significantly**. For example, SSIM should ideally decrease heavily when comparing an image with its flipped counterpart, while L1/L2 loss should increase. This guarantees the augmentations are actually scrambling spatial alignment as expected.

---

## Phase 4: Inference Verification Script

**Goal**: Verify that the production `DenoisePipeline` correctly loads models, processes images, and reduces noise.

**Implementation Steps**:
1.  **Create Script**: Create a new file `tests/verify_inference.py`.
2.  **Sample Data**: Load 10 clean, high-quality DICOMs (can be generated synthetically via `tests/gen_data.py`).
3.  **Noise Injection**: Add artificial Gaussian noise (e.g., $\sigma=0.05$) to the DICOM pixel data.
4.  **Inference Execution**:
    *   Instantiate `DenoisePipeline` from `src.pipeline` for a few core architectures (e.g., `unet`, `nafnet_small`). *(Note: initialize with random weights if no pretrained checkpoint is available).*
    *   Pass the noisy DICOMs through the `process_dicom` or `denoise_image` function.
    *   For now, assume no explicit sigma map is passed (or pass a uniform constant one if the architecture expects a 2-channel input).
5.  **Sanity Checks**:
    *   Assert that the output dimensions match the input dimensions.
    *   Assert that the output values are valid (no NaNs, and correctly rescaled to the 16-bit uint range `[0, 65535]`).

---

## Phase 5: Training Verification Script

**Goal**: Verify that a model can successfully overfit to a tiny dataset, ensuring that the backpropagation pipeline, loss aggregation, and optimizers are correctly wired.

**Implementation Steps**:
1.  **Create Script**: Create a new file `tests/verify_training.py`.
2.  **Minimal Environment**:
    *   Initialize a very small model (e.g., `unet` or a highly scaled-down `nafnet_xs`).
    *   Create a minimal `DataLoader` with just 2-4 pairs of `(noisy_input, clean_target)`.
3.  **Training Loop**:
    *   Execute 5-10 training epochs using the `Trainer` class (`src.trainer`).
4.  **Convergence Assertion**:
    *   Capture the loss at `Epoch 1` and the loss at `Epoch N`.
    *   Assert that the final loss is significantly strictly less than the initial loss.
    *   Assert that gradients are not `None` and do not contain NaNs during the backward pass.
