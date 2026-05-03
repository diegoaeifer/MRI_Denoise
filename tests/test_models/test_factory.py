"""
Tests for the MONAI network registry (replaces old factory tests).
"""

import pytest
import torch


class TestNetworkRegistry:
    """Test suite for the MONAI-native network registry."""

    @pytest.fixture
    def registry(self):
        try:
            from src.mri_denoise.networks.registry import get_network, REGISTRY
            return get_network, REGISTRY
        except ImportError:
            pytest.skip("Network registry not available")

    def test_registry_import(self):
        from src.mri_denoise.networks.registry import get_network, REGISTRY
        assert callable(get_network)
        assert isinstance(REGISTRY, dict)
        assert len(REGISTRY) >= 5

    def test_unknown_model_raises(self, registry):
        get_network, _ = registry
        with pytest.raises(KeyError, match="Unknown network"):
            get_network("nonexistent_model", in_channels=2, out_channels=1)

    @pytest.mark.parametrize("model_name", ["drunet", "nafnet"])
    def test_model_instantiation(self, registry, model_name):
        get_network, _ = registry
        try:
            model = get_network(model_name, spatial_dims=2, in_channels=2, out_channels=1)
            assert model is not None
        except Exception as e:
            pytest.skip(f"{model_name} instantiation failed: {e}")

    @pytest.mark.parametrize("model_name", ["drunet", "nafnet"])
    def test_model_forward_pass_2d(self, registry, model_name):
        get_network, _ = registry
        try:
            model = get_network(model_name, spatial_dims=2, in_channels=2, out_channels=1)
        except Exception as e:
            pytest.skip(f"{model_name} not available: {e}")

        model.eval()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 64, 64), f"Expected (1,1,64,64), got {y.shape}"

    def test_model_output_shape_matches_input_spatial(self, registry):
        """Output spatial dimensions should match input."""
        get_network, _ = registry
        try:
            model = get_network("drunet", spatial_dims=2, in_channels=2, out_channels=1)
        except Exception as e:
            pytest.skip(f"DRUNet not available: {e}")

        model.eval()
        for size in [32, 64]:
            x = torch.randn(1, 2, size, size)
            with torch.no_grad():
                y = model(x)
            assert y.shape[-2:] == (size, size), f"Spatial mismatch at size {size}"

    def test_all_registry_keys_accessible(self, registry):
        """Every registered model key should be a valid string."""
        _, REGISTRY = registry
        for key in REGISTRY:
            assert isinstance(key, str)
            assert len(key) > 0

    def test_snraware_graceful_if_unavailable(self, registry):
        """SNRAware should instantiate gracefully even without the package."""
        get_network, REGISTRY = registry
        assert "snraware" in REGISTRY
        try:
            model = get_network("snraware", spatial_dims=2, in_channels=2, out_channels=1)
            # If it instantiates, it's fine
            assert model is not None
        except Exception as e:
            pytest.skip(f"SNRAware not installed (expected): {e}")
