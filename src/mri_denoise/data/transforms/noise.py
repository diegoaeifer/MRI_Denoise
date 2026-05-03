"""
MONAI-native spatially-varying noise transform for MRI denoising.

This is the core custom transform that distinguishes the MRI denoising pipeline.
MONAI provides RandGaussianNoised and RandRicianNoised, but both use scalar σ.
This transform generates spatially-varying σ via grid interpolation.

The transform handles 2D (B, C, H, W) and 3D (B, C, H, W, D) natively.
"""

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms.transform import RandomizableTransform, MapTransform
from monai.utils import ensure_tuple, InterpolateMode
from typing import Dict, Hashable, List, Optional, Tuple, Union


class SpatiallyVaryingNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based transform that adds spatially-varying noise to an image.

    The noise is generated via bilinear (2D) or trilinear (3D) interpolation
    of a low-resolution random grid, ensuring smooth spatial variation.
    Both noisy image and σ-map are output.

    Args:
        keys: Image keys to apply noise to (e.g., ["image"])
        sigma_range: Tuple (sigma_min, sigma_max) for noise level, default (0.02, 0.3)
        grid_size: Grid resolution for noise interpolation, default 4 (coarse grid → smooth noise)
        noise_type: "gaussian" or "rician", default "gaussian"
        spatial_dims: 2 or 3, auto-detected from input if not specified
        prob: Probability of applying noise, default 1.0
    """

    def __init__(
        self,
        keys: Union[Hashable, List[Hashable]] = "image",
        sigma_range: Tuple[float, float] = (0.02, 0.3),
        grid_size: int = 4,
        noise_type: str = "gaussian",
        spatial_dims: Optional[int] = None,
        prob: float = 1.0,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.keys = ensure_tuple(keys)
        self.sigma_range = sigma_range
        self.grid_size = grid_size
        self.noise_type = noise_type.lower()
        self.spatial_dims = spatial_dims

        if self.noise_type not in ("gaussian", "rician"):
            raise ValueError(f"noise_type must be 'gaussian' or 'rician', got {self.noise_type}")

    def __call__(self, data: Dict) -> Dict:
        """
        Apply spatially-varying noise to all specified keys.

        Emits:
        - data[key]: noisy image
        - data[f"{key}_sigma_map"]: σ-map in [0, 1]
        """
        d = dict(data)

        # Randomize on first call
        if self.rand < self.prob:
            sigma_min, sigma_max = self.sigma_range
            sigma = self.R.uniform(sigma_min, sigma_max)

            for key in self.keys:
                if key not in d:
                    continue

                img = d[key]  # MetaTensor or Tensor, shape (C, H, W) or (C, H, W, D)
                img_np = img.numpy() if isinstance(img, torch.Tensor) else img

                # Auto-detect spatial_dims
                spatial_dims = self.spatial_dims or (3 if img_np.ndim == 4 else 2)

                # Generate noisy image and σ-map
                noisy_img, sigma_map = self._add_noise(
                    img_np, sigma, spatial_dims, self.noise_type
                )

                # Convert back to tensor, preserving MetaTensor metadata if present
                if isinstance(img, torch.Tensor):
                    d[key] = torch.from_numpy(noisy_img).type(img.dtype)
                    d[f"{key}_sigma_map"] = torch.from_numpy(sigma_map).float()
                else:
                    d[key] = noisy_img
                    d[f"{key}_sigma_map"] = sigma_map

        return d

    def _add_noise(
        self,
        img: np.ndarray,
        sigma: float,
        spatial_dims: int,
        noise_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add spatially-varying noise to image.

        Args:
            img: (C, H, W) or (C, H, W, D) numpy array
            sigma: noise level (0-1)
            spatial_dims: 2 or 3
            noise_type: "gaussian" or "rician"

        Returns:
            (noisy_img, sigma_map): both same shape as img, sigma_map float32
        """
        img = img.astype(np.float32)
        shape = img.shape[1:]  # (H, W) or (H, W, D)

        # Generate low-resolution random σ grid
        if spatial_dims == 2:
            grid_shape = (self.grid_size, self.grid_size)
        else:
            grid_shape = (self.grid_size, self.grid_size, self.grid_size)

        sigma_grid = self.R.uniform(0.5 * sigma, 1.5 * sigma, size=grid_shape)
        sigma_grid = np.clip(sigma_grid, 0, 1).astype(np.float32)

        # Interpolate grid to full resolution
        sigma_map = self._interp_grid(sigma_grid, shape, spatial_dims)

        # Generate noise with same shape as image
        if noise_type == "gaussian":
            noise = self.R.normal(0, 1, size=img.shape).astype(np.float32)
        else:  # rician
            # Rician noise: sqrt(I^2 + N1^2 + N2^2) where N1, N2 ~ N(0, σ^2)
            noise1 = self.R.normal(0, 1, size=img.shape).astype(np.float32)
            noise2 = self.R.normal(0, 1, size=img.shape).astype(np.float32)
            noise = np.sqrt(noise1**2 + noise2**2)  # envelope

        # Apply spatially-varying noise
        # Broadcast sigma_map to match image shape: (C, H, W) or (C, H, W, D)
        sigma_map_bc = np.expand_dims(sigma_map, axis=0)  # (1, H, W) or (1, H, W, D)
        noisy_img = img + sigma_map_bc * noise

        return noisy_img, sigma_map

    def _interp_grid(
        self,
        grid: np.ndarray,
        target_shape: Union[Tuple[int, int], Tuple[int, int, int]],
        spatial_dims: int,
    ) -> np.ndarray:
        """
        Interpolate low-res grid to target spatial shape using bilinear (2D) or trilinear (3D).

        Args:
            grid: (G, G) or (G, G, G) numpy array
            target_shape: (H, W) or (H, W, D)
            spatial_dims: 2 or 3

        Returns:
            Interpolated grid, shape target_shape, float32
        """
        # Convert to torch, add batch and channel dims
        grid_t = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)  # (1, 1, G, G) or (1, 1, G, G, G)

        # Interpolate to target shape
        mode = "bilinear" if spatial_dims == 2 else "trilinear"
        grid_t = F.interpolate(grid_t, size=target_shape, mode=mode, align_corners=False)

        # Remove batch/channel, convert back to numpy
        return grid_t.squeeze(0).squeeze(0).numpy().astype(np.float32)
