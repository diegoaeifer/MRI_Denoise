from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    RandAffined,
    RandZoomd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianSmoothd,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    CopyItemsd,
    ConcatItemsd,
)
from .noise import SpatiallyVaryingNoised


def build_train_transforms(cfg):
    """
    Builds the MONAI transform pipeline for training.
    Assumes config provides required parameters.
    """
    spatial_dims = cfg.get("spatial_dims", 2)
    patch_size = cfg.get("image_size", (256, 256))

    transforms = [
        LoadImaged(keys="image", reader="ITKReader"),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.05, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        # Spatial size standardization
        SpatialPadd(keys="image", spatial_size=patch_size),
        RandSpatialCropd(keys="image", roi_size=patch_size, random_size=False),
        # Copy original image to 'label' as the clean ground truth
        CopyItemsd(keys="image", times=1, names="label"),
        # Geometric (affects both image and label)
        RandAffined(
            keys=["image", "label"], prob=0.2, translate_range=(10, 10), mode="bilinear"
        ),
        RandZoomd(
            keys=["image", "label"],
            prob=0.1,
            min_zoom=0.9,
            max_zoom=1.1,
            mode="nearest",
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=(0, 1), prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        # Degradations (affects only 'image')
        RandAdjustContrastd(keys="image", gamma=(0.8, 1.2), prob=0.2),
        RandBiasFieldd(keys="image", prob=0.1),
        RandGaussianSmoothd(keys="image", prob=0.1),
        # Add Spatially Varying Noise (generates 'sigma')
        SpatiallyVaryingNoised(keys="image", spatial_dims=spatial_dims),
        # Clamp image to [0, 1]
        ScaleIntensityRanged(
            keys="image", a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True
        ),
        # Concat 'image' and 'sigma' for network input
        ConcatItemsd(keys=["image", "sigma"], name="image", dim=0),
    ]
    return Compose(transforms)


def build_val_transforms(cfg):
    """
    Builds the MONAI transform pipeline for validation.
    """
    spatial_dims = cfg.get("spatial_dims", 2)
    patch_size = cfg.get("image_size", (256, 256))

    transforms = [
        LoadImaged(keys="image", reader="ITKReader"),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRangePercentilesd(
            keys="image", lower=0.05, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        # Spatial size standardization
        SpatialPadd(keys="image", spatial_size=patch_size),
        RandSpatialCropd(keys="image", roi_size=patch_size, random_size=False),
        # Copy original image to 'label'
        CopyItemsd(keys="image", times=1, names="label"),
        # Add Spatially Varying Noise (deterministic or fixed seed if needed, but keeping random for now)
        SpatiallyVaryingNoised(keys="image", spatial_dims=spatial_dims),
        ScaleIntensityRanged(
            keys="image", a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True
        ),
        # Concat 'image' and 'sigma'
        ConcatItemsd(keys=["image", "sigma"], name="image", dim=0),
    ]
    return Compose(transforms)
