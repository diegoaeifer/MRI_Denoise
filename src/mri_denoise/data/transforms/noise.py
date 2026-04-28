import torch
import torch.nn.functional as F
from monai.transforms import MapTransform


class SpatiallyVaryingNoised(MapTransform):
    """
    Adds spatially varying noise to a given input tensor, while generating a matching noise map (sigma map).

    Compatible with 2D, 2.5D (slice stack), and 3D data.
    Input image should be (C, H, W) or (C, H, W, D).
    For 2.5D, input image has C > 1, representing adjacent slices.
    """

    def __init__(
        self,
        keys,
        sigma_range=(0.01, 0.1),
        grid_size=4,
        multiplier_range=(0.5, 3.0),
        noise_type="gaussian",
        spatial_dims=2,
        allow_missing_keys=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sigma_range = sigma_range
        self.grid_size = grid_size
        self.multiplier_range = multiplier_range
        self.noise_type = noise_type
        self.spatial_dims = spatial_dims

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]  # shape (C, H, W) or (C, H, W, D)

            # Determine base sigma
            sigma_base = torch.empty(1).uniform_(*self.sigma_range).item()

            # Create low-res modulation grid
            if self.spatial_dims == 3:
                mod_grid = torch.empty(
                    1,
                    1,
                    self.grid_size,
                    self.grid_size,
                    self.grid_size,
                    dtype=img.dtype,
                    device=img.device,
                ).uniform_(*self.multiplier_range)
                size = img.shape[1:]  # H, W, D
                mod_map = F.interpolate(
                    mod_grid, size=size, mode="trilinear", align_corners=False
                )
                mod_map = mod_map.squeeze(0)  # (1, H, W, D)
            else:
                # 2D or 2.5D (slice stacking)
                mod_grid = torch.empty(
                    1,
                    1,
                    self.grid_size,
                    self.grid_size,
                    dtype=img.dtype,
                    device=img.device,
                ).uniform_(*self.multiplier_range)
                size = img.shape[-2:]  # H, W
                mod_map = F.interpolate(
                    mod_grid, size=size, mode="bilinear", align_corners=False
                )
                mod_map = mod_map.squeeze(0)  # (1, H, W)

            # Sigma map
            sigma_map = mod_map * sigma_base

            # In 2.5D, the image has multiple channels, but the sigma map describes the noise
            # level. Usually, we assume the same noise distribution across adjacent slices if
            # they are acquired close in time/space, or we might need C independent sigma maps.
            # Here we apply the same sigma_map spatially, broadcast across C.

            if self.noise_type == "rician":
                noise1 = torch.randn_like(img) * sigma_map + img
                noise2 = torch.randn_like(img) * sigma_map
                noisy_img = torch.sqrt(noise1**2 + noise2**2)
            else:
                noise = torch.randn_like(img) * sigma_map
                noisy_img = img + noise

            d[key] = noisy_img
            # Add sigma map to dictionary, ensure it has shape (1, *spatial)
            d["sigma"] = sigma_map

        return d
