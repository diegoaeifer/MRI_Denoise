from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    raw_path: str = Field(..., description="Path to raw DICOM/NIfTI data")
    processed_path: str = Field(..., description="Path to processed data")
    patch_size: List[int] = Field(default=[256, 256], description="Patch size for training")

    normalization: Dict = Field(
        default={
            "type": "minmax_percentile",
            "percentile_min": 0.05,
            "percentile_max": 99.5,
            "bit_depth": 16,
        },
        description="Normalization parameters",
    )

    split_ratios: Dict[str, float] = Field(
        default={"train": 0.8, "val": 0.1, "test": 0.1}, description="Data split ratios"
    )

    augmentation: Dict = Field(
        default={
            "sigma_min": 0.05,
            "sigma_max": 0.10,
            "noise_type": "rician",
            "flip_prob": 0.5,
            "rotate_prob": 0.5,
            "ghosting_prob": 0.1,
            "gibbs_prob": 0.1,
            "motion_prob": 0.1,
        },
        description="Augmentation parameters",
    )

    num_workers: int = Field(default=4, description="Number of data loading workers")

    @field_validator("split_ratios")
    @classmethod
    def validate_split_ratios(cls, v):
        total = sum(v.values())
        if not (0.99 < total < 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return v

    @field_validator("augmentation")
    @classmethod
    def validate_augmentation(cls, v):
        if v["sigma_max"] < v["sigma_min"]:
            raise ValueError(
                f"sigma_max ({v['sigma_max']}) must be >= sigma_min ({v['sigma_min']})"
            )
        return v


class ModelConfig(BaseModel):
    type: str = Field(..., description="Model type (nafnet, drunet, unet, etc.)")
    in_channels: int = Field(default=2, description="Input channels (noisy_image + sigma_map)")
    out_channels: int = Field(default=1, description="Output channels")

    # Architecture-specific params
    width: Optional[int] = Field(default=64, description="Base width for models like NAFNet")
    depth: Optional[int] = Field(default=4, description="Depth of encoder/decoder")
    base_channels: Optional[int] = Field(default=64, description="Base channels for DRUNet")
    is_3d: bool = Field(default=False, description="Use 3D architecture")

    @field_validator("in_channels")
    @classmethod
    def validate_channels(cls, v):
        if v != 2:
            raise ValueError(f"in_channels must be 2 (image + sigma_map), got {v}")
        return v


class LossesConfig(BaseModel):
    weights: Dict[str, float] = Field(
        default={
            "l1": 1.0,
            "ssim": 1.0,
            "ms_ssim": 0.0,
            "psnr": 0.1,
            "haarpsi": 0.0,
            "epi": 0.0,
            "vgg": 0.0,
            "sure": 0.0,
            "lpips": 0.0,
            "dists": 0.0,
            "charbonnier": 0.0,
        },
        description="Loss weights for composite loss",
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v):
        valid_keys = {
            "l1",
            "ssim",
            "ms_ssim",
            "psnr",
            "haarpsi",
            "epi",
            "vgg",
            "sure",
            "lpips",
            "dists",
            "charbonnier",
        }
        for key in v.keys():
            if key not in valid_keys:
                raise ValueError(f"Unknown loss component: {key}")
            if v[key] < 0:
                raise ValueError(f"Loss weight must be non-negative, got {key}={v[key]}")
        return v


class TrainingConfig(BaseModel):
    epochs: int = Field(default=200, description="Number of training epochs")
    batch_size: int = Field(default=8, description="Batch size")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    optimizer: Literal["Adam", "AdamW", "SGD"] = Field(default="Adam", description="Optimizer type")
    scheduler: Optional[Literal["CosineAnnealing", "StepLR", "ReduceLROnPlateau"]] = Field(
        default="CosineAnnealing", description="Learning rate scheduler"
    )

    save_interval: int = Field(default=50, description="Save checkpoint every N epochs")
    gpu_id: int = Field(default=0, description="GPU device ID")
    seed: int = Field(default=42, description="Random seed")
    use_amp: bool = Field(default=False, description="Enable FP16 Automatic Mixed Precision")

    @field_validator("epochs", "batch_size")
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1:
            raise ValueError(f"Learning rate must be in (0, 1], got {v}")
        return v


class PipelineConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    losses: LossesConfig
    training: TrainingConfig

    output_dir: Optional[str] = Field(default="outputs", description="Output directory")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to checkpoint to load")
    test_mode: bool = Field(default=False, description="Run in test mode (fewer epochs)")

    def validate_on_load(self) -> bool:
        if self.model.is_3d and self.data.patch_size.__len__() != 3:
            raise ValueError("3D model requires 3D patch_size (e.g., [256, 256, 64])")
        if not self.model.is_3d and self.data.patch_size.__len__() != 2:
            raise ValueError("2D model requires 2D patch_size (e.g., [256, 256])")
        return True
