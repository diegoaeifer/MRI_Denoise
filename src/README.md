# Source Directory (`src/`)

This folder contains the core logic for the MRI Denoising pipeline. It is divided into sub-modules handling data, models, losses, and utilities.

## Core Execution Scripts

### `train.py`
The primary entry point for training models.
*   **Capabilities**: It initializes the model using the `get_model` factory, sets up the dataset and data loaders, configures the `CompositeLoss` and optimizers, and runs the training loop. It also integrates TensorBoard logging and model checkpointing.
*   **How to Use**: Run from the root directory. You can specify the model and configuration via arguments:
    ```bash
    python src/train.py --config configs/config_train.yaml --model drunet
    ```
    Use `--test` for a fast verification run.

### `inference.py`
The script used for evaluating a trained model on test data.
*   **Capabilities**: It loads the best checkpoint from `experiments/checkpoints/best_model.pth`, reconstructs the model architecture via the model factory, processes the test split defined in the data configs, and outputs visually restored images.
*   **How to Use**: Run from the root directory to perform inference on the default model (DRUNet) or specify one:
    ```bash
    python src/inference.py
    ```

## Sub-modules
*   [`src/data/`](data/README.md): Datasets, dataloaders, augmentations, and cleaning tools.
*   [`src/losses/`](losses/README.md): Custom loss functions for optimization.
*   [`src/models/`](models/README.md): Neural network definitions and the model factory.
*   [`src/utils/`](utils/README.md): Metrics and helpers.

## Antigravity Model Interaction
**STRICTLY OFF-LIMITS**. No files here (`train.py`, `pipeline.py`, etc.) will be modified. The Antigravity model will interface with `src/models/factory.py` (or bypass it entirely) by using an external wrapper script that injects the Antigravity PyTorch module into the standard trainer loop dynamically.
