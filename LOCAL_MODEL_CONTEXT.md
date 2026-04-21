# LOCAL MODEL CONTEXT
## CORE REPO FACTS
- **Domain**: MRI DICOM Denoising (Non-Blind)
- **Framework**: PyTorch (>=2.0.0), TorchIO, MONAI
- **Input Tensor Format**: 4D `(B, C, H, W)` typically `(B, 2, H, W)` [Image Channel, Sigma Noise Map]
- **Standard Normalization**: 16-bit percentile scaling (`np.percentile`, in-place operations)
- **Primary Metrics**: PSNR, MS-SSIM, HAARpsi (C=5.0, alpha=4.9)
- **Primary Loss**: CompositeLoss (Charbonnier, SSIM, PSNR, L1)

## ARCHITECTURAL RULES
- **Zero-Mod Integration**: External models must wrap via Adapter pattern, intercepting 2-channel input.
- **Hardware Limits (Test Env)**: 16GB VRAM, 32GB RAM -> Restrict batch sizes, use `pin_memory=True`.
- **Torch Safe Loading**: `torch.load` requires `weights_only=True`.
- **GPU Loop Optimizations**: Use `.detach()` for metric accumulation, avoid `.item()` in hot loops.
- **Warnings**: Suppress `piq` and `torchvision` init warnings locally via `warnings.catch_warnings()`.

## ANTIGRAVITY REQUIREMENTS
- **Interface**: Must accept `(B, 2, H, W)` and output `(B, 1, H, W)`.
- **Adapter Strategy**: Emulate `DeepinvPretrainedModel` ChannelAdapter. No `factory.py` modification.
- **Testing**: Isolated `pytest` suite. Mock `numpy` for missing standard libs, but NEVER mock `numpy` for data/math validation.
