# FMImaging MRI Denoise

A deep learning framework for MRI denoising, migrated to a MONAI-first architecture.

## Features

- **MONAI Integration:** Core data loading, augmentations, engines, and metrics utilize MONAI.
- **Spatially Varying Noise:** Custom `SpatiallyVaryingNoised` transform synthesizes realistic MRI noise distributions.
- **Advanced Architectures:** Custom versions of NAFNet, DRUNet, SCUNet, and VisNet via `mri_denoise/networks/registry.py`.
- **Composite Losses:** MONAI's SSIM and L1 combined with optional custom Edge Preservation Index (EPI) loss.
- **AMP Enabled:** Native mixed-precision training using MONAI `SupervisedTrainer` engines.

## Usage

### Train

```bash
python src/mri_denoise/train.py --config src/mri_denoise/configs/train.yaml
```

### Inference

```bash
python src/mri_denoise/inference.py --input_dir /path/to/noisy/scans --output_dir /path/to/output --ckpt /path/to/checkpoint.pth --network nafnet
```
