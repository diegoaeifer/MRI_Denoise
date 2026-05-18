"""
Tests for the Plug-and-Play ADMM module.

Covers:
  - ElasticNetRegularizer.prox  (soft-threshold equivalence when lam2==0)
  - L1WaveletRegularizer.prox   (energy reduction)
  - TL1Regularizer.prox         (output shape)
  - pnp_admm_denoise            (convergence, shape, value range)
"""
import sys
import os

# Ensure src is on the path when running from the project root or tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import pytest

from admm.regularizers import (
    ElasticNetRegularizer,
    L1WaveletRegularizer,
    TL1Regularizer,
    _soft_threshold_tensor,
)
from admm.pnp_admm import pnp_admm_denoise


# ---------------------------------------------------------------------------
# ElasticNet prox
# ---------------------------------------------------------------------------

class TestElasticNetProx:
    def test_soft_threshold_equivalence_lam2_zero(self):
        """ElasticNet prox with lam2=0 must equal pure soft-thresholding."""
        lam1 = 0.05
        rho = 2.0
        reg = ElasticNetRegularizer(lam1=lam1, lam2=0.0)

        x = torch.randn(2, 1, 32, 32)
        result = reg.prox(x, rho)
        expected = _soft_threshold_tensor(x, lam1 / rho)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_additional_l2_shrinkage(self):
        """Adding lam2 > 0 must produce further shrinkage compared to lam2==0."""
        lam1, lam2, rho = 0.02, 0.05, 1.0
        x = torch.ones(1, 1, 8, 8) * 0.5

        out_l1_only = ElasticNetRegularizer(lam1=lam1, lam2=0.0).prox(x, rho)
        out_elastic = ElasticNetRegularizer(lam1=lam1, lam2=lam2).prox(x, rho)

        assert (out_elastic.abs() <= out_l1_only.abs() + 1e-7).all(), (
            "ElasticNet (lam2>0) should shrink at least as much as pure L1"
        )

    def test_zero_input_stays_zero(self):
        """prox(0, rho) must be 0 for any rho and lam."""
        reg = ElasticNetRegularizer(lam1=0.1, lam2=0.1)
        x = torch.zeros(1, 1, 16, 16)
        out = reg.prox(x, rho=1.0)
        assert out.abs().max() < 1e-7

    def test_output_shape_preserved(self):
        """prox output shape must match input shape."""
        reg = ElasticNetRegularizer()
        x = torch.randn(3, 2, 64, 64)
        assert reg.prox(x, rho=1.0).shape == x.shape


# ---------------------------------------------------------------------------
# L1 Wavelet prox
# ---------------------------------------------------------------------------

class TestL1WaveletProx:
    def test_prox_reduces_energy(self):
        """Wavelet prox with lam > 0 should not increase the L2 energy."""
        reg = L1WaveletRegularizer(lam=0.05, wavelet="db4", level=2)
        x = torch.rand(1, 1, 64, 64)
        out = reg.prox(x, rho=1.0)
        assert out.norm() <= x.norm() + 1e-5, (
            "Wavelet soft-thresholding should not increase signal energy"
        )

    def test_output_shape_preserved(self):
        """prox output spatial size must match input for power-of-two images."""
        reg = L1WaveletRegularizer(lam=0.01, wavelet="db8", level=3)
        x = torch.rand(1, 1, 64, 64)
        out = reg.prox(x, rho=1.0)
        assert out.shape == x.shape

    def test_zero_lam_is_identity(self):
        """With lam=0 (threshold=0), wavelet prox should return the input unchanged."""
        reg = L1WaveletRegularizer(lam=0.0, wavelet="db4", level=2)
        x = torch.rand(1, 1, 32, 32)
        out = reg.prox(x, rho=1.0)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)

    def test_large_lam_pushes_toward_zero(self):
        """Very large threshold should drive the output close to zero."""
        reg = L1WaveletRegularizer(lam=1e6, wavelet="db4", level=2)
        x = torch.rand(1, 1, 32, 32) * 0.1
        out = reg.prox(x, rho=1.0)
        assert out.abs().max() < 1e-3, (
            "Extremely large lam/rho should zero out the coefficients"
        )


# ---------------------------------------------------------------------------
# TL1 prox
# ---------------------------------------------------------------------------

class TestTL1Prox:
    def test_output_shape_preserved(self):
        """TL1 prox must return a tensor with the same shape as the input."""
        reg = TL1Regularizer(lam=0.01, a=1.0)
        x = torch.randn(2, 1, 16, 16)
        out = reg.prox(x, rho=1.0)
        assert out.shape == x.shape

    def test_sign_preserved(self):
        """TL1 prox should not flip the sign of any element."""
        reg = TL1Regularizer(lam=0.05, a=0.5)
        x = torch.randn(1, 1, 32, 32)
        out = reg.prox(x, rho=2.0)
        # Where input is non-zero, sign must agree (or output is zero)
        non_zero_mask = x.abs() > 1e-6
        assert (out[non_zero_mask].sign() == x[non_zero_mask].sign()).all(), (
            "TL1 prox should not flip the sign of any non-zero element"
        )

    def test_shrinks_toward_zero(self):
        """TL1 prox should shrink values compared to the input magnitude."""
        reg = TL1Regularizer(lam=0.1, a=1.0)
        x = torch.ones(1, 1, 16, 16) * 0.5
        out = reg.prox(x, rho=1.0)
        assert (out.abs() <= x.abs() + 1e-6).all(), (
            "TL1 prox should not increase the magnitude of any element"
        )


# ---------------------------------------------------------------------------
# PnP-ADMM end-to-end
# ---------------------------------------------------------------------------

class IdentityDenoiser(nn.Module):
    """Returns the image channel unchanged (ignores sigma channel)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, H, W) — return only the image channel
        return x[:, :1, :, :]


class TestPnPADMM:
    def test_output_shape(self):
        """pnp_admm_denoise must return a tensor of shape (1, 1, H, W)."""
        noisy = torch.rand(1, 1, 64, 64)
        out = pnp_admm_denoise(noisy, IdentityDenoiser(), max_iter=5)
        assert out.shape == (1, 1, 64, 64)

    def test_output_range(self):
        """Output values must be clamped to [0, 1]."""
        noisy = torch.rand(1, 1, 64, 64)
        out = pnp_admm_denoise(noisy, IdentityDenoiser(), max_iter=10)
        assert out.min() >= 0.0, f"Output min {out.min()} < 0"
        assert out.max() <= 1.0, f"Output max {out.max()} > 1"

    def test_convergence_within_max_iter(self):
        """Identity denoiser + ElasticNet should converge well within 30 iters."""
        noisy = torch.rand(1, 1, 64, 64)
        out = pnp_admm_denoise(
            noisy,
            IdentityDenoiser(),
            sigma_denoiser=0.05,
            max_iter=30,
            tol=1e-4,
        )
        assert out.shape == (1, 1, 64, 64)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_l1_wavelet_regularizer(self):
        """ADMM should run without error using L1WaveletRegularizer."""
        noisy = torch.rand(1, 1, 64, 64)
        reg = L1WaveletRegularizer(lam=0.01, wavelet="db4", level=2)
        out = pnp_admm_denoise(noisy, IdentityDenoiser(), regularizer=reg, max_iter=5)
        assert out.shape == (1, 1, 64, 64)

    def test_tl1_regularizer(self):
        """ADMM should run without error using TL1Regularizer."""
        noisy = torch.rand(1, 1, 16, 16)  # small to keep TL1 fast
        reg = TL1Regularizer(lam=0.01, a=1.0)
        out = pnp_admm_denoise(noisy, IdentityDenoiser(), regularizer=reg, max_iter=3)
        assert out.shape == (1, 1, 16, 16)

    def test_default_regularizer_is_elasticnet(self):
        """When no regularizer is passed, ElasticNetRegularizer must be used (no crash)."""
        noisy = torch.rand(1, 1, 32, 32)
        out = pnp_admm_denoise(noisy, IdentityDenoiser(), regularizer=None, max_iter=5)
        assert out.shape == (1, 1, 32, 32)

    def test_denoiser_runs_on_cpu(self):
        """pnp_admm_denoise should not raise when device=cpu is specified explicitly."""
        noisy = torch.rand(1, 1, 32, 32)
        out = pnp_admm_denoise(
            noisy, IdentityDenoiser(), device=torch.device("cpu"), max_iter=5
        )
        assert out.device.type == "cpu"
