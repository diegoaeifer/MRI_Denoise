"""
MRI noise synthesis pipeline for training data augmentation.

Combines:
  1. Rician noise (always)
  2. Acceleration artifacts (GRAPPA 2x/3x, CS random, CAIPIRINHA 2x2) — optional
  3. Gibbs ringing (from transforms.py RandomGibbsRinging) — optional
  4. Temporal correlation for 2D+t sequences — optional
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class NoisePipelineConfig:
    """Configuration for the MRI noise synthesis pipeline.

    Attributes:
        sigma_range: (min, max) Rician noise standard deviation drawn uniformly.
        gibbs_prob: Probability in [0, 1] of applying Gibbs ringing.
        acceleration: Whether to apply k-space undersampling (GRAPPA-like).
        accel_factor: Undersampling factor (2 = keep every 2nd line, etc.).
        temporal: Whether to apply temporal noise correlation across slices.
    """
    sigma_range: Tuple[float, float] = (0.01, 0.10)
    gibbs_prob: float = 0.20
    acceleration: bool = False
    accel_factor: int = 2
    temporal: bool = False


class NoisePipeline:
    """
    Callable that takes a clean numpy image [H, W] (float32, [0, 1]) and
    returns (noisy: np.ndarray, sigma: float).

    The sigma is drawn uniformly from ``config.sigma_range`` on each call.
    Rician noise is always applied. Acceleration artifacts and Gibbs ringing
    are applied conditionally based on the config.

    Parameters
    ----------
    config : NoisePipelineConfig, optional
        Pipeline configuration. Defaults to ``NoisePipelineConfig()``.
    """

    def __init__(self, config: NoisePipelineConfig = NoisePipelineConfig()):
        self.config = config

    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply the full noise pipeline to a single clean image.

        Parameters
        ----------
        img : np.ndarray
            Clean float32 array of shape (H, W) with values in [0, 1].

        Returns
        -------
        noisy : np.ndarray
            Degraded image, same shape and dtype as ``img``.
        sigma : float
            The Rician sigma actually used.
        """
        img = img.astype(np.float32)

        # 1. Draw sigma uniformly from the configured range
        sigma = float(np.random.uniform(self.config.sigma_range[0],
                                        self.config.sigma_range[1]))

        noisy = img.copy()

        # 2. Optional: acceleration / k-space undersampling
        if self.config.acceleration:
            noisy = self._apply_acceleration_mask(noisy, self.config.accel_factor)

        # 3. Optional: Gibbs ringing
        if np.random.random() < self.config.gibbs_prob:
            noisy = self._apply_gibbs(noisy)

        # 4. Rician noise (always last so sigma is well-defined in image domain)
        noisy = self._add_rician(noisy, sigma)

        return noisy, sigma

    def _add_rician(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Add Rician noise: ``sqrt((img + n1)^2 + n2^2)`` where n1, n2 ~ N(0, sigma).

        Parameters
        ----------
        img : np.ndarray
            Input float32 array.
        sigma : float
            Standard deviation of the two independent Gaussian noise components.

        Returns
        -------
        np.ndarray
            Rician-corrupted image, same shape as ``img``.
        """
        n1 = np.random.normal(0, sigma, img.shape).astype(np.float32)
        n2 = np.random.normal(0, sigma, img.shape).astype(np.float32)
        return np.sqrt((img + n1) ** 2 + n2 ** 2)

    def _apply_acceleration_mask(self, img: np.ndarray, factor: int) -> np.ndarray:
        """Apply GRAPPA-like k-space undersampling along the phase-encode direction.

        Keeps every ``factor``-th line plus the central 24 lines (ACS region).
        All other k-space lines are zeroed. The result is the magnitude of the
        inverse FFT — this approximates aliasing / ghosting artifacts from
        parallel-imaging undersampling.

        Parameters
        ----------
        img : np.ndarray
            Float32 image of shape (H, W).
        factor : int
            Undersampling factor (e.g. 2 keeps every 2nd phase-encode line).

        Returns
        -------
        np.ndarray
            Magnitude image after masked inverse FFT, same shape as ``img``.
        """
        H, W = img.shape

        # Forward FFT and centre shift
        kspace = np.fft.fftshift(np.fft.fft2(img.astype(np.complex64)))

        # Build 1-D mask along phase-encode (row) direction
        mask_1d = np.zeros(H, dtype=bool)

        # Regularly sampled lines (every factor-th line)
        mask_1d[::factor] = True

        # ACS region: central 24 lines always included
        acs_half = 12  # 24 / 2
        cx = H // 2
        acs_start = max(0, cx - acs_half)
        acs_end = min(H, cx + acs_half)
        mask_1d[acs_start:acs_end] = True

        # Broadcast to 2-D and apply
        mask_2d = mask_1d[:, np.newaxis]  # (H, 1) — broadcast over columns
        kspace_masked = kspace * mask_2d

        # Inverse FFT → magnitude
        img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
        return img_rec.astype(np.float32)

    def _apply_gibbs(self, img: np.ndarray, truncation_pct: float = 0.05) -> np.ndarray:
        """Simulate Gibbs ringing by truncating the outer k-space.

        Keeps the central ``(1 - truncation_pct)`` fraction of k-space lines
        (in both dimensions) and zeros the rest, then returns the magnitude of
        the inverse FFT.

        Parameters
        ----------
        img : np.ndarray
            Float32 image of shape (H, W).
        truncation_pct : float, optional
            Fraction of outermost k-space lines to zero out (default 0.05 = 5 %).

        Returns
        -------
        np.ndarray
            Image with Gibbs ringing artifact, same shape as ``img``.
        """
        H, W = img.shape

        kspace = np.fft.fftshift(np.fft.fft2(img.astype(np.complex64)))

        # Number of lines to *keep* on each side of centre
        keep_h = int(np.round(H * (1.0 - truncation_pct) / 2.0))
        keep_w = int(np.round(W * (1.0 - truncation_pct) / 2.0))

        keep_h = max(1, keep_h)
        keep_w = max(1, keep_w)

        cx, cy = H // 2, W // 2

        mask = np.zeros((H, W), dtype=np.float32)
        mask[cx - keep_h: cx + keep_h, cy - keep_w: cy + keep_w] = 1.0

        kspace_trunc = kspace * mask

        img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_trunc)))
        return img_rec.astype(np.float32)


class TemporalNoisePipeline(NoisePipeline):
    """
    Extension of NoisePipeline that adds temporal noise correlation for 2D+t
    (cine / multi-frame) MRI sequences.

    Consecutive frames share a proportion of their noise drawn from the same
    realisation, creating inter-frame correlation while still sampling the
    noise level independently per call.

    Parameters
    ----------
    config : NoisePipelineConfig
        Pipeline configuration. ``config.temporal`` should be ``True``.
    temporal_corr : float, optional
        Correlation coefficient in [0, 1].  ``0`` = fully independent,
        ``1`` = identical noise across frames.  Default is ``0.5``.
    """

    def __init__(self,
                 config: NoisePipelineConfig = NoisePipelineConfig(),
                 temporal_corr: float = 0.5):
        super().__init__(config)
        self.temporal_corr = float(np.clip(temporal_corr, 0.0, 1.0))
        # Persistent noise fields reused across calls
        self._shared_n1: Optional[np.ndarray] = None
        self._shared_n2: Optional[np.ndarray] = None

    def reset(self):
        """Clear the shared noise state (call between independent volumes)."""
        self._shared_n1 = None
        self._shared_n2 = None

    def _add_rician(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Rician noise with optional temporal correlation.

        If a shared noise field exists (from a previous call on the same
        volume), the new noise is mixed with it proportionally to
        ``self.temporal_corr``.

        Parameters
        ----------
        img : np.ndarray
            Input float32 array of shape (H, W).
        sigma : float
            Standard deviation of the Gaussian noise components.

        Returns
        -------
        np.ndarray
            Rician-corrupted image with temporally correlated noise.
        """
        fresh_n1 = np.random.normal(0, sigma, img.shape).astype(np.float32)
        fresh_n2 = np.random.normal(0, sigma, img.shape).astype(np.float32)

        if self._shared_n1 is None or self._shared_n1.shape != img.shape:
            # First frame: no correlation possible — initialise shared state
            self._shared_n1 = fresh_n1
            self._shared_n2 = fresh_n2
        else:
            # Mix shared noise with fresh noise
            alpha = self.temporal_corr
            fresh_n1 = alpha * self._shared_n1 + np.sqrt(1.0 - alpha ** 2) * fresh_n1
            fresh_n2 = alpha * self._shared_n2 + np.sqrt(1.0 - alpha ** 2) * fresh_n2
            # Update shared state for the next frame
            self._shared_n1 = fresh_n1
            self._shared_n2 = fresh_n2

        return np.sqrt((img + fresh_n1) ** 2 + fresh_n2 ** 2)
