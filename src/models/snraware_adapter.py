import torch
import torch.nn as nn


class SNRAwareAdapter(nn.Module):
    """
    Adapter to integrate the SNRAware DenoisingModel into the MONAI-based pipeline.

    The SNRAware model normally expects complex data:
    Input: [B, C_in=3, T, H, W] (real, imag, gmap)
    Output: [B, C_out=2, T, H, W] (real, imag)

    In our MONAI pipeline, we process magnitude data with an estimated noise sigma map.
    Input: [B, 2, H, W] or [B, 2, H, W, D] (noisy_magnitude, noise_sigma_map)
    Output: [B, 1, H, W] or [B, 1, H, W, D] (denoised_magnitude)

    This adapter bridges the gap by:
    1. Initializing the SNRAware model with C_in=2 and C_out=1.
    2. Treating the `noise_sigma_map` from the pipeline identically to the `gmap`
       in the SNRAware logic, passing it as the second channel.
    3. Permuting and expanding dimensions to match the [B, C, T, H, W] format expected
       by SNRAware.
    """

    def __init__(self, config=None, D=16, H=64, W=64):
        super().__init__()

        try:
            from snraware.projects.mri.denoising.model import DenoisingModel
        except ImportError:
            raise ImportError(
                "The 'snraware' module is not installed or not in PYTHONPATH. "
                "Please install Microsoft SNRAware to use this model."
            )

        if config is None:
            # Create a default OmegaConf if none is provided
            from omegaconf import OmegaConf

            config_dict = {
                "backbone": {
                    "name": "unet",
                    "num_of_channels": 64,
                    "block_str": ["T1L1G1", "T1L1G1"],
                    "block": {
                        "cell": {
                            "window_size": [8, 8, 16],
                            "patch_size": [4, 4, 2],
                            "norm_mode": "layer",
                            "n_head": 64,
                        }
                    },
                }
            }
            config = OmegaConf.create(config_dict)

        # Initialize the SNRAware model with C_in=2 (magnitude + noise_map) and C_out=1 (magnitude)
        self.model = DenoisingModel(config=config, D=D, H=H, W=W, C_in=2, C_out=1)

    def forward(self, x):
        """
        Forward pass.
        Expects x: [B, 2, H, W] (2D) or [B, 2, H, W, D] (3D)
        where channel 0 is the noisy magnitude image and channel 1 is the noise sigma map.
        """
        original_ndim = x.ndim

        if original_ndim == 4:
            # Convert 2D [B, 2, H, W] to 3D with T=1: [B, 2, 1, H, W]
            x = x.unsqueeze(2)
        elif original_ndim == 5:
            # Convert 3D [B, 2, H, W, D] to SNRAware's [B, 2, D, H, W] (where T=D)
            x = x.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Expected 4D or 5D input, got {original_ndim}D")

        # The input x is now [B, 2, T, H, W].
        # In SNRAware, channel 0 is magnitude, channel 1 is the noise map (acting as the gmap).

        out = self.model(x)

        # The output `out` is [B, 1, T, H, W]

        if original_ndim == 4:
            # Revert back to 2D [B, 1, H, W]
            out = out.squeeze(2)
        elif original_ndim == 5:
            # Revert back to 3D [B, 1, H, W, D]
            out = out.permute(0, 1, 3, 4, 2)

        return out
