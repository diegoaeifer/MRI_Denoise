# Models Directory

This module contains the PyTorch implementations of the various deep learning architectures supported by the pipeline, alongside a factory for easy instantiation.

## Files & Capabilities

*   **`factory.py`**: Contains the `get_model(model_name, config)` function.
    *   **Capabilities**: Acts as the central hub for instantiating models. It reads the model name string and corresponding configuration from the YAML files to return the correct PyTorch model object.
    *   **How to Use**: Used directly in `train.py` and `inference.py`. For example: `model = get_model('nafnet', config['models'])`.
*   **`nafnet.py`**: Implementation of NAFNet (Nonlinear Activation Free Network).
    *   **Capabilities**: A state-of-the-art architecture optimized for image restoration. The code supports scaling into Small, Medium, and Large variants (up to 112M parameters) by adjusting block counts and channel widths.
*   **`drunet.py`**: Implementation of DRUNet.
    *   **Capabilities**: A classic architecture known for effective denoising via deep residual UNet structures.
*   **`scunet.py`**: Implementation of SCUNet.
    *   **Capabilities**: Swin-Conv-UNet, blending Convolutional and Transformer layers for capturing local and global features.
*   **`unet.py`**: Implementation of a standard UNet.
    *   **Capabilities**: A baseline encoder-decoder model used for comparative purposes and simpler tasks.

All model files are designed to accept a 2-channel input (Noisy Image + Sigma Map) to perform non-blind denoising. They are not run directly; they are accessed via `factory.py`.

## Antigravity Model Interaction
**STRICTLY OFF-LIMITS**. Do not add `antigravity.py` here. Do not modify `factory.py`. The Antigravity model will be defined in the external `antigravity_integration/` folder, wrapped in a class that mimics `DeepinvPretrainedModel` or `BaseMRIModel`, and passed to the trainer externally.
