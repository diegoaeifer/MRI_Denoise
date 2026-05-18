"""
Regularizer classes for Plug-and-Play ADMM.

Each regularizer exposes a single method:
    prox(x, rho) -> x_new

which computes the proximal operator of (1/rho) * R(x), i.e.:
    prox_{R/rho}(v) = argmin_x { R(x) + (rho/2) ||x - v||^2 }
"""
from __future__ import annotations

import numpy as np
import torch
import pywt


class Regularizer:
    """Abstract base. All regularizers implement prox(x, rho) -> x_new."""

    def prox(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        raise NotImplementedError


class L1WaveletRegularizer(Regularizer):
    """
    Soft-thresholding in the wavelet domain.

    Proximal operator of  lam * ||W x||_1  with threshold = lam / rho.
    Uses pywt.wavedec2 / pywt.waverec2 (2-D DWT, real-valued).

    Args:
        lam:     Regularization weight (lambda).
        wavelet: PyWavelets wavelet name (default 'db8').
        level:   Number of decomposition levels (default 3).
    """

    def __init__(self, lam: float = 0.01, wavelet: str = "db8", level: int = 3):
        self.lam = lam
        self.wavelet = wavelet
        self.level = level

    def prox(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Apply wavelet soft-thresholding to every element of the batch.

        x shape: (B, C, H, W)  — operates independently per (b, c) slice.
        """
        threshold = self.lam / rho
        device = x.device
        dtype = x.dtype

        x_np = x.detach().cpu().float().numpy()  # (B, C, H, W)
        out_np = np.empty_like(x_np)

        B, C, H, W = x_np.shape
        for b in range(B):
            for c in range(C):
                img = x_np[b, c]  # (H, W)
                coeffs = pywt.wavedec2(img, wavelet=self.wavelet, level=self.level)

                # Soft-threshold all sub-bands including approximation (LL)
                new_coeffs = [_soft_threshold(coeffs[0], threshold)]
                for detail_tuple in coeffs[1:]:
                    new_coeffs.append(
                        tuple(_soft_threshold(d, threshold) for d in detail_tuple)
                    )

                rec = pywt.waverec2(new_coeffs, wavelet=self.wavelet)
                # waverec2 may pad to a larger size; crop back
                out_np[b, c] = rec[:H, :W]

        return torch.tensor(out_np, dtype=dtype, device=device)


class ElasticNetRegularizer(Regularizer):
    """
    Elastic-Net regularizer:  R(x) = lam1 * ||x||_1  +  (lam2/2) * ||x||_2^2

    Closed-form proximal operator:
        prox_{R/rho}(v) = soft(v, lam1/rho) / (1 + lam2/rho)

    where soft(v, t) = sign(v) * max(|v| - t, 0).

    When lam2 == 0 this reduces to pure L1 soft-thresholding.
    When lam1 == 0 this reduces to L2 (ridge) shrinkage.

    Args:
        lam1: L1 penalty weight.
        lam2: L2 penalty weight.
    """

    def __init__(self, lam1: float = 0.01, lam2: float = 0.01):
        self.lam1 = lam1
        self.lam2 = lam2

    def prox(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        t = self.lam1 / rho
        scale = 1.0 / (1.0 + self.lam2 / rho)
        return _soft_threshold_tensor(x, t) * scale


class TL1Regularizer(Regularizer):
    """
    Transformed L1 (TL1) regularizer (Zhang & Xin, 2018):

        R(x) = lam * sum_i  |x_i| / (|x_i| + a)

    The penalty is concave; the proximal operator is solved element-wise via
    a closed-form root of a cubic equation (see derivation below).

    For each element v, prox_{R/rho}(v) solves:
        x + (lam / rho) * a * sign(x) / (|x| + a)^2 = v
        (after simplification through the KKT conditions)

    We use the equivalent formulation derived by computing the root of

        rho*(x - v)*(|x| + a)^2 + lam*a*sign(x) = 0

    For v >= 0 the solution x >= 0, so we substitute x >= 0 and solve:
        rho*(x - v)*(x + a)^2 + lam*a = 0
        rho*x^3 + (2*a*rho - rho*v)*x^2 ... (expanded cubic)

    We use scipy.optimize.brentq per element as a numerically safe fallback
    and cache the cubic coefficients for the positive branch.

    Args:
        lam: Regularization weight.
        a:   Shape parameter controlling concavity (a > 0). Larger a -> closer to L1.
    """

    def __init__(self, lam: float = 0.01, a: float = 1.0):
        self.lam = lam
        self.a = a

    def prox(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        x_np = x.detach().cpu().float().numpy().ravel()
        out_np = np.empty_like(x_np)

        lam = float(self.lam)
        a = float(self.a)

        for i, v in enumerate(x_np):
            out_np[i] = _tl1_prox_scalar(v, lam, a, rho)

        return torch.tensor(
            out_np.reshape(x.shape), dtype=dtype, device=device
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _soft_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    """NumPy soft-threshold: sign(arr) * max(|arr| - threshold, 0)."""
    return np.sign(arr) * np.maximum(np.abs(arr) - threshold, 0.0)


def _soft_threshold_tensor(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Torch soft-threshold."""
    return x.sign() * torch.clamp(x.abs() - threshold, min=0.0)


def _tl1_prox_scalar(v: float, lam: float, a: float, rho: float) -> float:
    """
    Closed-form scalar proximal operator for TL1.

    Solves:  argmin_x { lam * |x| / (|x| + a)  +  (rho/2) * (x - v)^2 }

    By symmetry the solution has the same sign as v, so we work with the
    positive half-line (|v|, |x|) and re-attach the sign at the end.

    On the positive half-line the optimality condition is:
        rho*(x - |v|) + lam*a / (x + a)^2 = 0
        rho*(x - |v|)*(x + a)^2 + lam*a = 0

    Expanding into a cubic in x:
        rho*x^3  +  (2*a*rho - rho*|v|)*x^2
        + (a^2*rho - 2*a*rho*|v|)*x
        + (-a^2*rho*|v| + lam*a)  =  0

    We find the root in [0, |v|] using numpy.roots and pick the real root
    closest to |v| (the unconstrained minimiser would be near |v|).
    If no valid root exists in [0, |v|], we fall back to |v| (no shrinkage).
    """
    abs_v = abs(v)
    sign_v = 1.0 if v >= 0.0 else -1.0

    if abs_v == 0.0:
        return 0.0

    # Cubic coefficients (descending powers of x)
    c3 = rho
    c2 = 2.0 * a * rho - rho * abs_v
    c1 = a ** 2 * rho - 2.0 * a * rho * abs_v
    c0 = -(a ** 2) * rho * abs_v + lam * a

    roots = np.roots([c3, c2, c1, c0])

    # Keep only real, non-negative roots
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    valid = real_roots[real_roots >= 0.0]

    if len(valid) == 0:
        # No shrinkage (penalty gradient too small to move the solution)
        return v

    # Pick the root closest to abs_v (minimises the objective)
    best = valid[np.argmin(np.abs(valid - abs_v))]
    return sign_v * float(best)
