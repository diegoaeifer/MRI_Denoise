# FMImaging MRI Denoising Pipeline 🧠🔬

A robust, non-blind deep learning pipeline for denoising MRI DICOM images. This project supports multiple state-of-the-art architectures, spatially varying noise modeling, and high-throughput data cleaning.

## 🌟 Capabilities & Features

*   **Non-Blind Denoising**: Efficient 2-channel input (Noisy Image + estimated Sigma Map) for noise-adaptive restoration.
*   **Multi-Model Support**: Integrated factory for **NAFNet**, **DRUNet**, **SCUNet**, and **UNet**.
*   **NAFNet Scaling**: Small, Medium, and Large variants (up to 112M parameters) with width-32 optimization.
*   **Massive Data Cleaning**: Specialized script to scan and auto-delete uninformative "flat" images in batches.
*   **MRI-Specific Augmentation**: 
    *   Interpolation-free ±90° discrete rotations to prevent aliasing.
    *   Vertical and Horizontal flipping.
    *   Anisotropy simulation for slice resolution handling.
*   **16-bit Normalization**: Advanced percentile-based scaling tailored for high-dynamic-range MRI data.
*   **Composite Loss**: Balanced optimization using Charbonnier, SSIM, PSNR, and L1.
*   **Inference & Evaluation**: End-to-end evaluation pipeline with inference script.

## 🛠️ Installation

```bash
git clone <repo-url>
cd FMImaging_MRI_Denoise
pip install -r requirements.txt
```

## 🚀 Usage Guide

### 1. Data Cleaning
Remove "mostly black" images that hinder model convergence:
```bash
python src/data/check_background.py --data_path data/IXI --delete --batch_size 1000
```

### 2. Training the Model
Launch a full NAFNet-17M run with Charbonnier loss:
```bash
python src/train.py --config configs/config_nafnet_production.yaml --model nafnet
```

### 3. Inference and Evaluation
Run inference on an existing model to evaluate its performance:
```bash
python src/inference.py
```

### 4. Test Mode (Quick Verification)
To run a fast trial with 1000 images and 10 epochs using specific test data:
```bash
python src/train.py --test --model nafnet
```

## 📂 Project Structure

This project is organized into several modules. **Detailed documentation for each sub-folder can be found within the respective directories:**

*   **`configs/`**: Contains YAML files for configuring models, data loading, and training parameters. [See configs/README.md](configs/README.md)
*   **`src/`**: The core source code. [See src/README.md](src/README.md)
    *   **`src/data/`**: Data loading, dataset classes, and MRI specific augmentations.
    *   **`src/losses/`**: Custom loss functions (e.g., Composite, Auxiliary).
    *   **`src/models/`**: Neural network architectures (NAFNet, UNet, etc.) and model factory.
    *   **`src/utils/`**: Utility scripts, including metric calculations.
*   **`tests/`**: Unit tests and data generation scripts for verification. [See tests/README.md](tests/README.md)

## 🧪 Quick Test Scripts
At the root of the project, there are a couple of helpful scripts:
*   **`run_trials.py`**: A batch testing script designed to run trial training iterations across multiple model architectures (NAFNet, DRUNet, SCUNet, UNet) sequentially. It is useful for verifying that all models can initialize and complete basic training steps without errors.
*   **`test_inplace_float32.py`**: A small script demonstrating and validating the in-place clipping and 16-bit normalization logic used in our dataset pre-processing, ensuring memory efficiency and correct `float32` bounds handling.

*(Note: `Colab_Training_MRI.ipynb` is currently put on hold and not the primary method for training.)*

## 📊 Monitoring
Logs and sample images are saved to `experiments/`. Use TensorBoard to visualize metrics:
```bash
tensorboard --logdir experiments/logs
```

## 📄 License
MIT
