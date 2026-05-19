import pytest
from pydantic import ValidationError
from src.config import (
    DataConfig,
    ModelConfig,
    LossesConfig,
    TrainingConfig,
    PipelineConfig,
    get_defaults,
)


class TestDataConfig:
    def test_valid_data_config(self):
        config = DataConfig(
            raw_path="data/IXI",
            processed_path="data/processed",
        )
        assert config.raw_path == "data/IXI"
        assert config.num_workers == 4

    def test_invalid_split_ratios_sum(self):
        with pytest.raises(ValidationError):
            DataConfig(
                raw_path="data/IXI",
                processed_path="data/processed",
                split_ratios={"train": 0.5, "val": 0.3, "test": 0.3},  # Sum = 1.1
            )

    def test_sigma_max_less_than_sigma_min(self):
        with pytest.raises(ValidationError):
            DataConfig(
                raw_path="data/IXI",
                processed_path="data/processed",
                augmentation={"sigma_min": 0.1, "sigma_max": 0.05},
            )


class TestModelConfig:
    def test_valid_model_config(self):
        config = ModelConfig(
            type="drunet",
            in_channels=2,
            out_channels=1,
        )
        assert config.type == "drunet"
        assert config.in_channels == 2

    def test_invalid_in_channels(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                type="drunet",
                in_channels=3,  # Should be 2
                out_channels=1,
            )

    def test_3d_model_config(self):
        config = ModelConfig(
            type="rician_net_3d",
            in_channels=2,
            out_channels=1,
            is_3d=True,
        )
        assert config.is_3d is True


class TestLossesConfig:
    def test_valid_losses_config(self):
        config = LossesConfig()
        assert config.weights["l1"] == 1.0
        assert config.weights["ssim"] == 1.0

    def test_invalid_loss_key(self):
        with pytest.raises(ValidationError):
            LossesConfig(weights={"l1": 1.0, "unknown_loss": 0.5})

    def test_negative_loss_weight(self):
        with pytest.raises(ValidationError):
            LossesConfig(weights={"l1": -1.0})

    def test_custom_loss_weights(self):
        config = LossesConfig(
            weights={
                "l1": 0.0,
                "ssim": 1.0,
                "charbonnier": 0.5,
            }
        )
        assert config.weights["ssim"] == 1.0
        assert config.weights["charbonnier"] == 0.5


class TestTrainingConfig:
    def test_valid_training_config(self):
        config = TrainingConfig(epochs=100, batch_size=8)
        assert config.epochs == 100
        assert config.batch_size == 8

    def test_invalid_epochs(self):
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=2.0)  # Should be <= 1.0

    def test_optimizer_choices(self):
        for optimizer in ["Adam", "AdamW", "SGD"]:
            config = TrainingConfig(optimizer=optimizer)
            assert config.optimizer == optimizer

    def test_invalid_optimizer(self):
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer="InvalidOptimizer")


class TestPipelineConfig:
    def test_valid_pipeline_config(self, dummy_config):
        assert isinstance(dummy_config, PipelineConfig)
        assert dummy_config.data.raw_path is not None
        assert dummy_config.model.in_channels == 2

    def test_3d_model_requires_3d_patch_size(self):
        config = PipelineConfig(
            data=DataConfig(
                raw_path="data/IXI",
                processed_path="data/processed",
                patch_size=[256, 256, 64],  # 3D
            ),
            model=ModelConfig(type="rician_net", in_channels=2, is_3d=True),
            losses=LossesConfig(),
            training=TrainingConfig(),
        )
        config.validate_on_load()  # Should not raise

    def test_2d_model_with_3d_patch_size_fails(self):
        config = PipelineConfig(
            data=DataConfig(
                raw_path="data/IXI",
                processed_path="data/processed",
                patch_size=[256, 256, 64],  # 3D
            ),
            model=ModelConfig(type="drunet", in_channels=2, is_3d=False),
            losses=LossesConfig(),
            training=TrainingConfig(),
        )
        with pytest.raises(ValueError):
            config.validate_on_load()

    def test_default_config_is_valid(self):
        config = get_defaults()
        config.validate_on_load()  # Should not raise
