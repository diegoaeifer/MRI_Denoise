# Configs Directory

This directory contains YAML configuration files that control various aspects of the MRI Denoising pipeline. These configs decouple hyperparameters from code, allowing you to easily run experiments without modifying scripts.

## Files

*   **`config_data.yaml`**: Configures dataset paths, split ratios, 16-bit min/max percentile normalization parameters, patch sizes, and complex MRI-specific data augmentation parameters (such as rotation probabilities, noise multipliers, ghosting, gibbs ringing, and blur probabilities).
*   **`config_model.yaml`**: Defines architecture-specific hyperparameters. For example, it specifies the number of encoder/decoder blocks, width, and middle blocks for `nafnet` (including small, medium, and large variants), `drunet`, `scunet`, and standard `unet`.
*   **`config_train.yaml`**: Contains training loop configurations such as the number of epochs, batch size, learning rate, optimizer selection (Adam, AdamW), learning rate scheduler, save intervals, and GPU ID allocation.
*   **`config_debug.yaml`**: A specialized, scaled-down configuration file used primarily for rapid testing and debugging (e.g., fewer epochs, smaller batch sizes).
*   **`config_nafnet_production.yaml`**: A complete configuration file tailored for full-scale production training of the NAFNet model.

## How to Use
These files are automatically loaded by `src/train.py` and `src/inference.py`. By default, the scripts merge `config_train.yaml`, `config_data.yaml`, and `config_model.yaml`.

You can override this and pass a specific master configuration file (like the production one) using the `--config` flag:
```bash
python src/train.py --config configs/config_nafnet_production.yaml --model nafnet
```

## Antigravity Model Interaction
**STRICTLY OFF-LIMITS (Do Not Touch)**. The Antigravity model must NOT modify these core configurations. Instead, the Antigravity integration suite will programmatically load these configs in-memory using `yaml.safe_load`, and dynamically override keys (e.g., `model_name`, memory parameters) at runtime within isolated adapter scripts.
