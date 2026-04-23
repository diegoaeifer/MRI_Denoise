import os
from typing import List, Dict, Optional, Any
import yaml
from .schemas import PipelineConfig, DataConfig, ModelConfig, LossesConfig, TrainingConfig


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def get_defaults() -> PipelineConfig:
    """Return default valid configuration."""
    return PipelineConfig(
        data=DataConfig(
            raw_path="data/IXI",
            processed_path="data/processed",
        ),
        model=ModelConfig(
            type="drunet",
            in_channels=2,
            out_channels=1,
            is_3d=False,
        ),
        losses=LossesConfig(),
        training=TrainingConfig(),
    )


def load_and_validate(config_paths: Optional[List[str]] = None) -> PipelineConfig:
    """Load config files, merge, and validate against schema.

    Args:
        config_paths: List of config file paths to load and merge.
                     If None, returns defaults.

    Returns:
        Validated PipelineConfig object.

    Raises:
        FileNotFoundError: If a config file doesn't exist.
        ValueError: If config validation fails.
    """
    # Start with defaults
    config_dict = {
        "data": get_defaults().data.model_dump(),
        "model": get_defaults().model.model_dump(),
        "losses": get_defaults().losses.model_dump(),
        "training": get_defaults().training.model_dump(),
    }

    # Load and merge each config file
    if config_paths:
        for path in config_paths:
            if path is None:
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(f"Config file not found: {path}")
            loaded = load_yaml(path)
            config_dict = merge_configs(config_dict, loaded)

    # Validate using Pydantic schemas
    try:
        pipeline_config = PipelineConfig(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            losses=LossesConfig(**config_dict.get("losses", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            output_dir=config_dict.get("output_dir", "outputs"),
            checkpoint_path=config_dict.get("checkpoint_path"),
            test_mode=config_dict.get("test_mode", False),
        )
        pipeline_config.validate_on_load()
        return pipeline_config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")
