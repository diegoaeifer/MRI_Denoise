"""MRI Denoising & Super-Resolution Benchmark

Methods:
  - L2 (Tikhonov, FFT closed-form)
  - L1 TV (ADMM)
  - ElasticNet (ADMM, dynamic lambda schedule)
  - TL1 (Transformed L1, ADMM with Newton proximal)
  - WCRR (RED Accelerated Gradient Descent)
  - GSD-DRUNet (PnP-GD + tissue mask)
Super-resolution: WCRR vs Lanczos
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Optional imports
try:
    from skimage.filters import threshold_otsu
    _HAS_OTSU = True
except ImportError:
    _HAS_OTSU = False

NOISE_LEVELS = [0.05, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_rician_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Rician noise: sqrt((image+n1)^2 + n2^2) where n1,n2 ~ N(0,sigma^2)."""
    rng = np.random.default_rng()
    n1 = rng.normal(0, sigma, image.shape)
    n2 = rng.normal(0, sigma, image.shape)
    noisy = np.sqrt((image + n1) ** 2 + n2 ** 2)
    return noisy.astype(np.float64)


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def _grad(x: np.ndarray) -> List[np.ndarray]:
    """Finite differences gradient. Returns list of arrays, one per spatial dim."""
    gs = []
    for d in range(x.ndim):
        gs.append(np.roll(x, -1, axis=d) - x)
    return gs


def _div(*gs: np.ndarray) -> np.ndarray:
    """Negative divergence (adjoint of grad)."""
    result = np.zeros_like(gs[0])
    for d, g in enumerate(gs):
        # backward diff: g - roll(g, +1, axis=d)
        result += -(g - np.roll(g, 1, axis=d))
    return result


def _lap_kernel(shape: Tuple[int, ...]) -> np.ndarray:
    """Laplacian eigenvalues in frequency domain (positive values)."""
    kernel = np.zeros(shape)
    for d, n in enumerate(shape):
        freq = np.fft.fftfreq(n) * 2 * np.pi
        # reshape for broadcasting
        reshape = [1] * len(shape)
        reshape[d] = n
        freq = freq.reshape(reshape)
        kernel = kernel + 2 * (np.cos(freq) - 1)
    # negate to get positive values
    return -kernel


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_l2(y: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """Tikhonov: x* = IFFT[ FFT[y] / (1 + lam * |xi|^2) ]"""
    lap = _lap_kernel(y.shape)  # positive
    Y = np.fft.fftn(y)
    denom = np.maximum(1.0 + lam * lap, 1e-10)
    X = Y / denom
    return np.real(np.fft.ifftn(X))


def solve_l1_admm(y: np.ndarray, lam: float = 0.05, rho: float = 1.0, n_iter: int = 50) -> np.ndarray:
    """TV-L1 via ADMM."""
    ndim = y.ndim
    lap = _lap_kernel(y.shape)
    Y = np.fft.fftn(y)

    x = y.copy()
    z = [np.zeros_like(y) for _ in range(ndim)]
    u = [np.zeros_like(y) for _ in range(ndim)]

    for _ in range(n_iter):
        # x-update: FFT closed-form
        div_zu = _div(*[z[d] - u[d] for d in range(ndim)])
        denom = np.maximum(1.0 + rho * lap, 1e-10)
        X = (Y + rho * np.fft.fftn(div_zu)) / denom
        x = np.real(np.fft.ifftn(X))

        # z-update: soft threshold
        gx = _grad(x)
        for d in range(ndim):
            v = gx[d] + u[d]
            threshold = lam / rho
            z[d] = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

        # u-update (reuse gx computed above)
        for d in range(ndim):
            u[d] = u[d] + gx[d] - z[d]

    return x


def solve_elasticnet_admm(
    y: np.ndarray,
    lam1: float = 0.05,
    lam2: float = 0.05,
    rho: float = 1.0,
    n_iter: int = 50,
    dynamic: bool = True,
) -> np.ndarray:
    """ElasticNet TV via ADMM with dynamic lambda schedule."""
    ndim = y.ndim
    lap = _lap_kernel(y.shape)
    Y = np.fft.fftn(y)

    x = y.copy()
    z = [np.zeros_like(y) for _ in range(ndim)]
    u = [np.zeros_like(y) for _ in range(ndim)]

    for k in range(n_iter):
        lam1_k = lam1 * (1 - k / n_iter) if dynamic else lam1
        lam2_k = lam2 * (k / n_iter) if dynamic else lam2

        # x-update
        div_zu = _div(*[z[d] - u[d] for d in range(ndim)])
        denom = np.maximum(1.0 + (rho + 2 * lam2_k) * lap, 1e-10)
        X = (Y + rho * np.fft.fftn(div_zu)) / denom
        x = np.real(np.fft.ifftn(X))

        # z-update: soft threshold with dynamic lam1_k
        gx = _grad(x)
        for d in range(ndim):
            v = gx[d] + u[d]
            threshold = lam1_k / rho
            z[d] = np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

        # u-update (reuse gx computed above)
        for d in range(ndim):
            u[d] = u[d] + gx[d] - z[d]

    return x


def _tl1_prox(v: np.ndarray, gamma: float, a: float) -> np.ndarray:
    """TL1 proximal operator via Newton method."""
    abs_v = np.abs(v)
    # Initial guess
    t = np.maximum(abs_v - gamma * (a + 1) / a, 0.0)

    for _ in range(3):
        at = a + t
        f = gamma * (a + 1) * a / (at ** 2) + t - abs_v
        fp = 1.0 - 2.0 * gamma * (a + 1) * a / (at ** 3)
        # Avoid division by near-zero
        fp = np.where(np.abs(fp) < 1e-10, 1e-10, fp)
        t = t - f / fp
        t = np.maximum(t, 0.0)

    return np.sign(v) * t


def solve_tl1_admm(
    y: np.ndarray,
    lam: float = 0.05,
    a: float = 1.0,
    rho: float = 1.0,
    n_iter: int = 50,
) -> np.ndarray:
    """TL1 TV via ADMM."""
    ndim = y.ndim
    lap = _lap_kernel(y.shape)
    Y = np.fft.fftn(y)

    x = y.copy()
    z = [np.zeros_like(y) for _ in range(ndim)]
    u = [np.zeros_like(y) for _ in range(ndim)]

    for _ in range(n_iter):
        # x-update
        div_zu = _div(*[z[d] - u[d] for d in range(ndim)])
        denom = np.maximum(1.0 + rho * lap, 1e-10)
        X = (Y + rho * np.fft.fftn(div_zu)) / denom
        x = np.real(np.fft.ifftn(X))

        # z-update: TL1 prox
        gx = _grad(x)
        for d in range(ndim):
            v = gx[d] + u[d]
            z[d] = _tl1_prox(v, lam / rho, a)

        # u-update (reuse gx computed above)
        for d in range(ndim):
            u[d] = u[d] + gx[d] - z[d]

    return x


def solve_wcrr(
    y: np.ndarray,
    lam: float = 0.1,
    sigma: float = 0.1,
    n_iter: int = 100,
    step_size: float = 0.5,
) -> np.ndarray:
    """WCRR via RED + Accelerated Gradient Descent (Nesterov)."""
    blur_sigma = sigma * 10

    def D(x):
        return scipy.ndimage.gaussian_filter(x, sigma=blur_sigma)

    x = y.copy()
    x_prev = x.copy()
    t = 1.0

    for k in range(n_iter):
        g = (x - y) + lam * (x - D(x))
        x_new = x - step_size * g
        x_new = np.clip(x_new, 0, 1)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        x_prev_old = x_prev.copy()
        x_prev = x.copy()
        x = x_new + (t - 1) / t_new * (x_new - x_prev_old)
        t = t_new

    return np.clip(x, 0, 1)


def compute_tissue_mask(image: np.ndarray) -> np.ndarray:
    """Otsu threshold. Returns boolean mask (True = tissue)."""
    if _HAS_OTSU:
        try:
            thresh = threshold_otsu(image)
        except Exception:
            thresh = np.mean(image)
    else:
        thresh = np.mean(image)
    return image > thresh


def solve_gsd_drunet(
    y: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma_noise: float = 0.1,
    tau: float = 0.5,
    n_iter: int = 30,
) -> np.ndarray:
    """GSD-DRUNet approximation (PnP-GD + tissue mask)."""
    blur_sigma = sigma_noise * 5

    def D_sigma(x):
        return scipy.ndimage.gaussian_filter(x, sigma=blur_sigma)

    if mask is None:
        mask = compute_tissue_mask(y)

    mask_f = mask.astype(np.float64)

    x = y.copy()
    for k in range(n_iter):
        grad_f = x - y
        x_denoised = D_sigma(x - tau * grad_f)
        x = mask_f * x_denoised + (1 - mask_f) * y

    return np.clip(x, 0, 1)


# ---------------------------------------------------------------------------
# SR helpers
# ---------------------------------------------------------------------------

def _match_shape(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Crop or pad arr to exactly target_shape."""
    slices = tuple(slice(0, s) for s in target_shape)
    cropped = arr[slices]
    pad = [(0, max(0, t - c)) for t, c in zip(target_shape, cropped.shape)]
    return np.pad(cropped, pad, mode='edge')

def downscale(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downscale image by factor using cubic spline."""
    return scipy.ndimage.zoom(image, 1.0 / factor, order=3)


def upscale_lanczos(image_lr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Upscale 2D image to target_shape using Lanczos filter."""
    if image_lr.ndim == 2:
        arr_uint8 = (np.clip(image_lr, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(arr_uint8)
        pil_resized = pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
        return np.array(pil_resized).astype(np.float32) / 255.0
    else:
        # 3D: apply slice-by-slice on first axis
        slices = []
        for i in range(image_lr.shape[0]):
            sl = upscale_lanczos(image_lr[i], target_shape[1:])
            slices.append(sl)
        result = np.stack(slices, axis=0)
        # Ensure first axis matches
        if result.shape[0] != target_shape[0]:
            zoom_factors = [target_shape[0] / result.shape[0]] + [1.0] * (result.ndim - 1)
            result = scipy.ndimage.zoom(result, zoom_factors, order=1)
        return result


def upscale_wcrr(
    image_lr: np.ndarray,
    target_shape: Tuple[int, ...],
    lam: float = 0.05,
    sigma: float = 0.05,
    n_iter: int = 100,
) -> np.ndarray:
    """Upscale using WCRR SR (RED + AGD)."""
    # Initialize with Lanczos
    x = upscale_lanczos(image_lr, target_shape).astype(np.float64)

    factor = target_shape[0] // image_lr.shape[0]
    factor_inv = image_lr.shape[0] / target_shape[0]
    blur_sigma = sigma * 10
    step_size = 0.1

    def D(arr):
        return scipy.ndimage.gaussian_filter(arr, sigma=blur_sigma)

    def A(arr):
        out = scipy.ndimage.zoom(arr, factor_inv, order=1)
        # Ensure shape matches image_lr
        slices = tuple(slice(0, s) for s in image_lr.shape)
        if out.shape != image_lr.shape:
            # pad or crop
            result = np.zeros(image_lr.shape)
            crop = tuple(slice(0, min(out.shape[d], image_lr.shape[d])) for d in range(out.ndim))
            result[crop] = out[crop]
            out = result
        return out

    def AT(v):
        out = scipy.ndimage.zoom(v, 1.0 / factor_inv, order=1)
        # Ensure shape matches target_shape
        if out.shape != target_shape:
            result = np.zeros(target_shape)
            crop = tuple(slice(0, min(out.shape[d], target_shape[d])) for d in range(out.ndim))
            result[crop] = out[crop]
            out = result
        return out

    x_prev = x.copy()
    t = 1.0

    for k in range(n_iter):
        lr_approx = A(x)
        residual = lr_approx - image_lr
        grad_data = AT(residual)
        # Ensure grad_data shape matches x
        if grad_data.shape != x.shape:
            result = np.zeros_like(x)
            crop = tuple(slice(0, min(grad_data.shape[d], x.shape[d])) for d in range(x.ndim))
            result[crop] = grad_data[crop]
            grad_data = result

        grad_data = _match_shape(grad_data, x.shape)
        g = grad_data + lam * (x - D(x))
        x_new = x - step_size * g
        x_new = np.clip(x_new, 0, 1)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        x_prev_old = x_prev.copy()
        x_prev = x.copy()
        x = x_new + (t - 1) / t_new * (x_new - x_prev_old)
        t = t_new

    return _match_shape(np.clip(x, 0, 1), target_shape)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Compute PSNR, SSIM, NRMSE."""
    psnr = peak_signal_noise_ratio(original, reconstructed, data_range=1.0)

    if original.ndim == 2:
        ssim = structural_similarity(original, reconstructed, data_range=1.0)
    else:
        ssim = structural_similarity(
            original,
            reconstructed,
            data_range=1.0,
            win_size=3,
            channel_axis=None,
        )

    nrmse = np.linalg.norm(original - reconstructed) / (np.linalg.norm(original) + 1e-8)
    return {"psnr": psnr, "ssim": ssim, "nrmse": nrmse}


# ---------------------------------------------------------------------------
# Worker / Coordinator agents
# ---------------------------------------------------------------------------

DISPATCH = {
    "l2": lambda y, **kw: solve_l2(y, **kw),
    "l1": lambda y, **kw: solve_l1_admm(y, **kw),
    "elasticnet": lambda y, **kw: solve_elasticnet_admm(y, **kw),
    "tl1": lambda y, **kw: solve_tl1_admm(y, **kw),
    "wcrr": lambda y, **kw: solve_wcrr(y, **kw),
    "gsd_drunet": lambda y, **kw: solve_gsd_drunet(y, **kw),
}


def worker_agent(task: dict) -> dict:
    """Run one denoising method, save .npz, return metrics dict."""
    method = task["method"]
    if method not in DISPATCH:
        raise ValueError(f"Unknown method: {method!r}. Valid: {list(DISPATCH)}")
    y = task["noisy"]
    clean = task.get("clean")
    kwargs = task.get("kwargs", {})
    output_dir = task.get("output_dir", "benchmark_results")

    fn = DISPATCH[method]
    denoised = fn(y, **kwargs)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{method}_sigma{task.get('sigma', 'NA')}.npz")
    np.savez(out_path, denoised=denoised, noisy=y)

    result = {"method": method, "sigma": task.get("sigma", None)}
    if clean is not None:
        metrics = compute_metrics(clean, denoised)
        result.update(metrics)

    return result


def coordinator_agent(
    image: np.ndarray,
    output_dir: str,
    noise_levels: List[float],
    methods: List[str],
) -> None:
    """Fan out worker_agent calls, write results.csv and .done sentinel."""
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for sigma in noise_levels:
        noisy = add_rician_noise(image, sigma)
        for method in methods:
            tasks.append({
                "method": method,
                "noisy": noisy,
                "clean": image,
                "sigma": sigma,
                "output_dir": os.path.join(output_dir, f"sigma_{sigma}"),
            })

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker_agent, t) for t in tasks]
        try:
            rows = [f.result() for f in concurrent.futures.as_completed(futures)]
        except Exception as e:
            print(f"Worker failed: {e}", file=sys.stderr)
            raise
        results.extend(rows)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    # Sentinel
    with open(os.path.join(output_dir, ".done"), "w") as f:
        json.dump({"status": "done", "n_results": len(results)}, f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MRI Denoising Benchmark")
    parser.add_argument("input", help="Path to input image (.npy, .npz, .nii, etc.)")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=NOISE_LEVELS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["l2", "l1", "elasticnet", "tl1", "wcrr", "gsd_drunet"],
    )
    parser.add_argument("--slice-idx", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load image
    input_path = Path(args.input)
    if input_path.suffix == ".npy":
        image = np.load(str(input_path))
    elif input_path.suffix == ".npz":
        data = np.load(str(input_path))
        image = data[list(data.keys())[0]]
    elif input_path.suffix in (".nii", ".gz"):
        import nibabel as nib
        image = nib.load(str(input_path)).get_fdata()
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")

    # Normalize
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    if args.slice_idx is not None and image.ndim == 3:
        image = image[args.slice_idx]

    coordinator_agent(image, args.output_dir, args.noise_levels, args.methods)
    print(f"Done. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
