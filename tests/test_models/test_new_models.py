"""Tests for the three newly integrated denoising models.

Tests cover:
  - Factory registration (model name recognised by get_model)
  - Output shape: (B, 1, H, W) for (B, 2, H, W) input
  - No NaN / Inf in output
  - Gradient flow

CDLNet and Restore-RWKV tests are skipped if the corresponding package
cannot be imported (repo not cloned or CUDA extension unavailable).
AstroDenoiser is a pure-PyTorch implementation and should always pass.

Note: The factory's __init__.py performs eager imports of all model modules.
When optional dependencies (einops, deepinv, etc.) are not installed the
``src.models.factory`` import raises ImportError.  In that case factory-level
tests are skipped, but wrapper-level tests for AstroDenoiser still run
because they import the wrapper directly.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add MRI_Denoise root to sys.path so "src.models.*" imports work regardless
# of where pytest is invoked from.
_MRI_DENOISE_ROOT = Path(__file__).parent.parent.parent
if str(_MRI_DENOISE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MRI_DENOISE_ROOT))

# Also add src/models directly so we can import wrappers without triggering
# src/models/__init__.py (which eagerly imports the full factory chain and
# requires optional deps like einops that may not be installed).
_MODELS_DIR = _MRI_DENOISE_ROOT / "src" / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(b: int = 1, h: int = 32, w: int = 32) -> torch.Tensor:
    """Return a (B, 2, H, W) tensor mimicking (noisy_image, sigma_map)."""
    img = torch.randn(b, 1, h, w)
    sigma = torch.rand(b, 1, h, w) * 0.1
    return torch.cat([img, sigma], dim=1)


def _minimal_config() -> dict:
    """Minimal factory config dict (mirrors config_model.yaml structure)."""
    return {
        "common": {
            "in_channels": 2,
            "out_channels": 1,
        },
    }


def _try_import_factory():
    """Return (get_model, _MODEL_BUILDERS) or (None, None) if not importable."""
    try:
        from src.models.factory import get_model, _MODEL_BUILDERS  # noqa: PLC0415
        return get_model, _MODEL_BUILDERS
    except Exception:
        return None, None


def _get_model_via_factory(model_name: str):
    """Import get_model and construct *model_name* from the factory."""
    get_model, _ = _try_import_factory()
    if get_model is None:
        pytest.skip(
            "src.models.factory not fully importable "
            "(optional deps like einops may be missing)"
        )
    return get_model(model_name, _minimal_config())


# ---------------------------------------------------------------------------
# Per-model skip guards (lazy import checks)
# ---------------------------------------------------------------------------

def _cdlnet_available() -> bool:
    """True when CDLNET repo is importable."""
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent / "FMImaging_MRI_Denoise" / "CDLNET"
    if not repo.exists():
        return False
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        import model.net  # noqa: F401
        return True
    except Exception:
        return False


def _restore_rwkv_available() -> bool:
    """True when Restore-RWKV repo is importable (needs CUDA extension)."""
    from pathlib import Path
    repo = Path(__file__).parent.parent.parent / "FMImaging_MRI_Denoise" / "Restore-RWKV"
    if not repo.exists():
        return False
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        from model.Restore_RWKV import Restore_RWKV  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------

class TestFactoryRegistration:
    """Verify that all three model names are known to get_model."""

    @pytest.mark.parametrize("model_name", ["cdlnet", "restore_rwkv", "astro_denoiser"])
    def test_model_name_in_builders(self, model_name):
        """Model name should appear in _MODEL_BUILDERS."""
        _, _MODEL_BUILDERS = _try_import_factory()
        if _MODEL_BUILDERS is None:
            pytest.skip(
                "src.models.factory not fully importable "
                "(optional deps like einops may be missing)"
            )
        assert model_name in _MODEL_BUILDERS, (
            f"'{model_name}' missing from _MODEL_BUILDERS"
        )

    def test_invalid_name_raises(self):
        """Unknown name should raise ValueError."""
        get_model, _ = _try_import_factory()
        if get_model is None:
            pytest.skip(
                "src.models.factory not fully importable "
                "(optional deps like einops may be missing)"
            )
        with pytest.raises(ValueError, match="not implemented"):
            get_model("nonexistent_model_xyz", _minimal_config())


# ---------------------------------------------------------------------------
# AstroDenoiser — always available (pure PyTorch, no optional deps)
# ---------------------------------------------------------------------------

class TestAstroDenoiser:
    """Tests for the pure-PyTorch AstroDenoiserWrapper.

    Uses direct wrapper import to avoid depending on the full factory
    import chain (which requires optional deps like einops).
    """

    def _model(self):
        from astro_denoiser_wrapper import AstroDenoiserWrapper  # noqa: PLC0415
        return AstroDenoiserWrapper(in_channels=2)

    def test_instantiation(self):
        model = self._model()
        assert model is not None

    @pytest.mark.parametrize("b,h,w", [(1, 32, 32), (2, 64, 64), (1, 48, 64)])
    def test_output_shape(self, b, h, w):
        model = self._model().eval()
        x = _make_input(b, h, w)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (b, 1, h, w), (
            f"Expected ({b}, 1, {h}, {w}), got {out.shape}"
        )

    def test_no_nan_or_inf(self):
        model = self._model().eval()
        x = _make_input(1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_gradient_flow(self):
        model = self._model().train()
        x = _make_input(1, 32, 32).requires_grad_(True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None, "No gradient flowed back to input"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), (
            "Gradients are all zero"
        )

    def test_parameter_count_positive(self):
        model = self._model()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has no parameters"


# ---------------------------------------------------------------------------
# CDLNet — skipped if repo not cloned / importable
# ---------------------------------------------------------------------------

class TestCDLNet:
    """Tests for CDLNetWrapper (requires CDLNET repo to be cloned)."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if not _cdlnet_available():
            pytest.skip(
                "CDLNET not importable — clone with: "
                "git clone https://github.com/nikopj/CDLNET-OJSP "
                "FMImaging_MRI_Denoise/CDLNET"
            )

    def _model(self):
        from cdlnet_wrapper import CDLNetWrapper  # noqa: PLC0415
        # K=3, adaptive=False for sanity tests — K=20+adaptive need pretrained weights
        return CDLNetWrapper(in_channels=2, K=3, adaptive=False)

    def test_instantiation(self):
        model = self._model()
        assert model is not None

    @pytest.mark.parametrize("b,h,w", [(1, 32, 32), (2, 64, 64)])
    def test_output_shape(self, b, h, w):
        model = self._model().eval()
        x = _make_input(b, h, w)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (b, 1, h, w), (
            f"Expected ({b}, 1, {h}, {w}), got {out.shape}"
        )

    def test_no_nan_or_inf(self):
        model = self._model().eval()
        x = _make_input(1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_gradient_flow(self):
        model = self._model().train()
        x = _make_input(1, 32, 32).requires_grad_(True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None, "No gradient flowed back to input"


# ---------------------------------------------------------------------------
# Restore-RWKV — skipped if repo not cloned or CUDA extension unavailable
# ---------------------------------------------------------------------------

class TestRestoreRWKV:
    """Tests for RestoreRWKVWrapper (requires CUDA extension compilation)."""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        if not _restore_rwkv_available():
            pytest.skip(
                "Restore-RWKV not importable (CUDA extension may be missing) — "
                "clone with: git clone https://github.com/Yaziwel/Restore-RWKV "
                "FMImaging_MRI_Denoise/Restore-RWKV"
            )

    def _model(self):
        from restore_rwkv_wrapper import RestoreRWKVWrapper  # noqa: PLC0415
        return RestoreRWKVWrapper(in_channels=2)

    def test_instantiation(self):
        model = self._model()
        assert model is not None

    @pytest.mark.parametrize("b,h,w", [(1, 32, 32), (2, 64, 64)])
    def test_output_shape(self, b, h, w):
        model = self._model().eval()
        x = _make_input(b, h, w)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (b, 1, h, w), (
            f"Expected ({b}, 1, {h}, {w}), got {out.shape}"
        )

    def test_no_nan_or_inf(self):
        model = self._model().eval()
        x = _make_input(1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_gradient_flow(self):
        model = self._model().train()
        x = _make_input(1, 32, 32).requires_grad_(True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None, "No gradient flowed back to input"


# ---------------------------------------------------------------------------
# Parametric cross-model shape test (skips gracefully per model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["cdlnet", "restore_rwkv", "astro_denoiser"])
def test_shape(model_name):
    """Output shape is (1, 1, 32, 32) for (1, 2, 32, 32) input.

    CDLNet and Restore-RWKV are skipped if their repos/extensions are unavailable.
    AstroDenoiser is always available (pure PyTorch).
    """
    if model_name == "cdlnet" and not _cdlnet_available():
        pytest.skip("CDLNet repo not available")
    if model_name == "restore_rwkv" and not _restore_rwkv_available():
        pytest.skip("Restore-RWKV repo not available (CUDA extension)")

    # Use direct wrapper imports so tests run even if the full factory chain
    # can't be loaded (e.g. einops missing).
    if model_name == "cdlnet":
        from cdlnet_wrapper import CDLNetWrapper  # noqa: PLC0415
        model = CDLNetWrapper(in_channels=2, K=3, adaptive=False).eval()
    elif model_name == "restore_rwkv":
        from restore_rwkv_wrapper import RestoreRWKVWrapper  # noqa: PLC0415
        model = RestoreRWKVWrapper(in_channels=2).eval()
    else:  # astro_denoiser
        from astro_denoiser_wrapper import AstroDenoiserWrapper  # noqa: PLC0415
        model = AstroDenoiserWrapper(in_channels=2).eval()

    x = _make_input(b=1, h=32, w=32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 32, 32), (
        f"[{model_name}] Expected (1,1,32,32), got {out.shape}"
    )
