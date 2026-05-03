"""
MONAI-native transform pipeline for MRI denoising.

Strategy:
1. Use MONAI dict-transforms where available
2. Keep only custom SpatiallyVaryingNoised (core innovation)
3. Build Compose pipeline that emits (image, label, sigma_map) dicts
4. Final ConcatItemsd merges image + sigma_map → 2-channel tensor
"""

from .noise import SpatiallyVaryingNoised

__all__ = ["SpatiallyVaryingNoised", "build_train_transforms", "build_val_transforms"]


def build_train_transforms(config):
    """
    Build training-time MONAI transform pipeline.

    Config should have:
    - data.image_size: [H, W] or [H, W, D]
    - data.augmentation: dict with aug params
    - losses.weights: dict (needed to know which losses will consume the data)

    Returns:
        monai.transforms.Compose with dict-transforms
    """
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        ScaleIntensityRangePercentilesd,
        CropForegroundd,
        RandAffined,
        RandAdjustContrastd,
        RandBiasFieldd,
        RandGaussianSmoothd,
        RandFlipd,
        RandRotate90d,
        CopyItemsd,
        ConcatItemsd,
        ToTensord,
    )

    spatial_dims = config.get("spatial_dims", 2)

    # Build transform list
    transforms_list = [
        # Load and normalize
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.05,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # Copy clean image to label before noise
        CopyItemsd(keys="image", times=1, names="label"),
        # Spatial transformations
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandAffined(
            keys=["image", "label"],
            spatial_dims=spatial_dims,
            translate_range=(0.1, 0.1) if spatial_dims == 2 else (0.1, 0.1, 0.05),
            prob=0.5,
        ),
        RandRotate90d(keys=["image", "label"], spatial_dims=spatial_dims, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_dims=spatial_dims, prob=0.5),
        # Intensity augmentations
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        RandBiasFieldd(keys=["image"], prob=0.1),
        RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)),
        # Core: spatially-varying noise (MRI-specific)
        SpatiallyVaryingNoised(
            keys=["image"],
            sigma_range=tuple(config.get("data", {}).get("augmentation", {}).get("sigma_range", (0.02, 0.1))),
            grid_size=config.get("data", {}).get("augmentation", {}).get("grid_size", 4),
            noise_type=config.get("data", {}).get("augmentation", {}).get("noise_type", "gaussian"),
            spatial_dims=spatial_dims,
            prob=1.0,
        ),
        # Merge 2-channel: image (noisy) + sigma_map
        ConcatItemsd(keys=["image", "image_sigma_map"], name="image", dim=0),
        # Convert to tensor
        ToTensord(keys=["image", "label"]),
    ]

    return Compose(transforms_list)


def build_val_transforms(config):
    """
    Build validation-time transform pipeline (no augmentation, no noise).

    Returns:
        monai.transforms.Compose with dict-transforms
    """
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        ScaleIntensityRangePercentilesd,
        ToTensord,
    )

    transforms_list = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.05,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ToTensord(keys=["image"]),
    ]

    return Compose(transforms_list)
