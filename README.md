# FMImaging MRI Denoising Pipeline 🧠🔬

A robust, non-blind deep learning pipeline for denoising MRI DICOM images. This project supports multiple state-of-the-art architectures, spatially varying noise modeling, and high-throughput data cleaning.

## 🌟 Key Features

*   **Non-Blind Denoising**: Efficient 2-channel input (Noisy Image + estimated Sigma Map) for noise-adaptive restoration.
*   **Multi-Model Support**: Integrated factory for **NAFNet**, **DRUNet**, **SCUNet**, and **UNet**.
*   **NAFNet Scaling**: Small, Medium, and Large variants (up to 112M parameters) with width-32 optimization.
*   **Massive Data Cleaning**: Specialized script to scan and auto-delete uninformative "flat" images in batches (scanned 300k+ files in ~20 min).
*   **MRI-Specific Augmentation**: 
    *   Interpolation-free ±90° discrete rotations to prevent aliasing.
    *   Vertical and Horizontal flipping.
    *   Anisotropy simulation for slice resolution handling.
*   **16-bit Normalization**: Advanced percentile-based scaling tailored for high-dynamic-range MRI data.
*   **Composite Loss**: Balanced optimization using Charbonnier, SSIM, PSNR, and L1.

## 🛠️ Installation

```bash
git clone <repo-url>
cd FMImaging_MRI_Denoise
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Data Cleaning
Remove "mostly black" images that hinder model convergence:
```bash
python src/data/check_background.py --data_path data/IXI --delete --batch_size 1000
```

### 2. Production Training
Launch a full NAFNet-17M run with Charbonnier loss:
```bash
python src/train.py --config configs/config_nafnet_production.yaml --model nafnet
```

### 3. Test Mode (Quick Verification)
To run a fast trial with 1000 images and 10 epochs using specific test data:
```bash
python src/train.py --test --model nafnet
```

### 4. Model Variants
| Model | Name Hint | Blocks (Enc/Mid/Dec) | Params |
| :--- | :--- | :--- | :--- |
| NAFNet-Small | `nafnet_small` | [2,2,4,8] / 12 / [2,2,2,2] | 29M |
| NAFNet-Medium | `nafnet_medium` | [4,4,8,16] / 24 / [4,4,4,4] | 57M |
| NAFNet-Large | `nafnet_large` | [8,8,16,32] / 48 / [8,8,8,8] | 112M |

## 📊 Monitoring
Logs and sample images are saved to `experiments/`. Use TensorBoard to visualize metrics:
```bash
tensorboard --logdir experiments/logs
```

## 📄 License
MIT
