"""
MRI Denoising Benchmark — Pretrained Models (GPU)
==================================================
Models  : WCRR-2D (pretrained), GSD-DRUNet (pretrained), WCRR-3D (pretrained)
Protocol: GenRegBench nmAPG + L2 / L1 / ElasticNet data fidelity
Sequences: T1, T2, PD, MRA  (IXI dataset, middle 2-D slice + 3-D crop)
"""

from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import torchcde
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import deepinv as dinv

# ─── device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── paths ────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(__file__).parent / "weights"
IXI_DIR = Path("C:/projetos/Datasets/IXI/all")
OUT_DIR = Path("C:/projetos/benchmark_results/pretrained")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = [
    ("T1",  IXI_DIR / "IXI015-HH-1258-T1.nii.gz",    128),
    ("T2",  IXI_DIR / "IXI015-HH-1258-T2.nii.gz",    128),
    ("PD",  IXI_DIR / "IXI015-HH-1258-PD.nii.gz",    128),
    ("MRA", IXI_DIR / "IXI016-Guys-0697-MRA.nii.gz",  256),
]

NOISE_LEVELS = [0.05, 0.1, 0.2, 0.3]

# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════════════

class ZeroMean(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=(1, 2, 3), keepdim=True)


class WCRR2D(dinv.optim.Prior):
    """Weakly-Convex Reconstruction Regularizer (2-D).  GenRegBench."""
    def __init__(self, sigma=0.1, weak_convexity=1.0,
                 nb_channels=(1, 4, 8, 64), filter_sizes=(5, 5, 5)):
        super().__init__()
        self.nb_filters  = nb_channels[-1]
        self.in_channels = nb_channels[0]
        fsize = sum(filter_sizes) - len(filter_sizes) + 1

        self.filters = nn.Sequential(*[
            nn.Conv2d(nb_channels[i], nb_channels[i+1], filter_sizes[i],
                      padding=filter_sizes[i]//2, bias=False)
            for i in range(len(filter_sizes))
        ])
        P.register_parametrization(self.filters[0], "weight", ZeroMean())

        self.dirac = torch.zeros(nb_channels[0], nb_channels[0],
                                 2*fsize-1, 2*fsize-1)
        self.dirac[:, :, fsize-1, fsize-1] = 1.0

        self.scaling = nn.Parameter(
            torch.log(torch.tensor(2.0) / sigma) * torch.ones(1, self.nb_filters, 1, 1))
        self.beta     = nn.Parameter(torch.tensor(4.0))
        self.weak_cvx = weak_convexity

    def _smooth_l1(self, x):
        return torch.clip(x**2, 0, 1)/2 + torch.clip(x.abs(), 1) - 1.0

    def _grad_smooth_l1(self, x):
        return torch.clip(x, -1.0, 1.0)

    def _conv_lip(self):
        imp = self.filters(self.dirac.to(next(self.parameters()).device))
        for f in reversed(self.filters):
            imp = F.conv_transpose2d(imp, f.weight, padding=f.padding)
        return torch.fft.fft2(imp, s=[256, 256]).abs().max()

    def _conv(self, x):
        return self.filters(x / self._conv_lip().sqrt())

    def _conv_T(self, x):
        x = x / self._conv_lip().sqrt()
        for f in reversed(self.filters):
            x = F.conv_transpose2d(x, f.weight, padding=f.padding)
        return x

    def grad(self, x, *args, **kwargs):
        expanded = False
        if x.shape[1] == 1 and self.in_channels > 1:
            x = x.tile(1, self.in_channels, 1, 1); expanded = True
        g = self._conv(x) * torch.exp(self.scaling)
        g = (self._grad_smooth_l1(torch.exp(self.beta)*g)
             - self._grad_smooth_l1(g) * self.weak_cvx)
        g = g * torch.exp(-self.scaling)
        g = self._conv_T(g)
        return g.sum(1, keepdim=True) if expanded else g

    def _apply(self, fn):
        self.dirac = fn(self.dirac); return super()._apply(fn)


# ── 3-D WCRR ──────────────────────────────────────────────────────────────────

class LinearSpline(nn.Module):
    def __init__(self, N=16, K=12, sigma_min=0.01, sigma_max=0.1, eps=1e-5):
        super().__init__()
        self.N, self.K, self.eps = N, K, float(eps)
        self.register_buffer("t_knots", torch.linspace(sigma_min, sigma_max, K))
        self.s_at_knots = nn.Parameter(torch.zeros(1, K, N))

    def forward(self, sigma: torch.Tensor):
        sigma = sigma.view(-1)
        coeffs = torchcde.linear_interpolation_coeffs(self.s_at_knots, t=self.t_knots)
        s_vals = torchcde.LinearInterpolation(coeffs).evaluate(sigma).squeeze(0)
        alphas = torch.exp(s_vals) / (sigma.view(-1, 1) + self.eps)
        return alphas.view(len(sigma), self.N, 1, 1, 1)


class ZeroMean3D(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=(1, 2, 3, 4), keepdim=True)


class WCRR3D(dinv.optim.Prior):
    """Weakly-Convex Reconstruction Regularizer (3-D).  wcrr-noncartesian-3d-mri."""
    def __init__(self, weak_convexity=1.0, nb_channels=(2, 4, 8, 16),
                 filter_sizes=(5, 5, 5), rotations=True):
        super().__init__()
        self.nb_filters  = nb_channels[-1]
        self.weak_cvx    = weak_convexity
        self.rotations   = rotations
        fsize = sum(filter_sizes) - len(filter_sizes) + 1

        self.filters = nn.Sequential(*[
            nn.Conv3d(nb_channels[i], nb_channels[i+1], filter_sizes[i],
                      padding=filter_sizes[i]//2, bias=False)
            for i in range(len(filter_sizes))
        ])
        P.register_parametrization(self.filters[0], "weight", ZeroMean3D())

        sz   = 2*fsize - 1
        in_c = self.filters[0].in_channels
        self.dirac = torch.zeros(1, in_c, sz, sz, sz)
        self.dirac[0, :, fsize-1, fsize-1, fsize-1] = 1.0

        self.scaling = LinearSpline(N=self.nb_filters, K=12)
        self.beta    = nn.Parameter(torch.tensor(4.0))

    def _smooth_l1(self, x):
        return torch.clip(x**2, 0, 1)/2 + torch.clip(x.abs(), 1) - 1.0

    def _grad_smooth_l1(self, x):
        return torch.clip(x, -1.0, 1.0)

    def _conv_lip(self):
        imp = self.filters(self.dirac.to(next(self.parameters()).device))
        for f in reversed(self.filters):
            imp = F.conv_transpose3d(imp, f.weight, padding=f.padding)
        return torch.fft.fftn(imp.float()).abs().max().to(imp.dtype)

    def _conv(self, x):
        return self.filters(x / self._conv_lip().sqrt())

    def _conv_T(self, x):
        x = x / self._conv_lip().sqrt()
        for f in reversed(self.filters):
            x = F.conv_transpose3d(x, f.weight, padding=f.padding)
        return x

    def _grad_R(self, x, scale_sp):
        beta_sp = torch.exp(self.beta)
        g = self._conv(x) * scale_sp
        g = self._grad_smooth_l1(beta_sp*g) - self._grad_smooth_l1(g)*self.weak_cvx
        return self._conv_T(g / scale_sp)

    def grad(self, x, sigma):
        scale_sp = self.scaling(sigma)
        if self.rotations:
            xDH = torch.rot90(x, 1, (-3, -2))
            xDW = torch.rot90(x, 1, (-3, -1))
            xHW = torch.rot90(x, 1, (-2, -1))
            g = (self._grad_R(x, scale_sp)
                 + torch.rot90(self._grad_R(xDH, scale_sp), -1, (-3, -2))
                 + torch.rot90(self._grad_R(xDW, scale_sp), -1, (-3, -1))
                 + torch.rot90(self._grad_R(xHW, scale_sp), -1, (-2, -1)))
            return g / 4
        return self._grad_R(x, scale_sp)

    def _apply(self, fn):
        self.dirac = fn(self.dirac); return super()._apply(fn)


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_wcrr2d():
    model = WCRR2D(sigma=0.1, weak_convexity=1.0, nb_channels=(1, 4, 8, 64))
    sd = torch.load(WEIGHTS_DIR / "WCRR_gray.pt", map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing}")
    return model.eval().to(DEVICE)


def load_wcrr3d():
    model = WCRR3D(weak_convexity=1.0, nb_channels=(2, 4, 8, 32), filter_sizes=(3, 3, 3))
    sd = torch.load(WEIGHTS_DIR / "WCRR_3d.pt", map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing}")
    return model.eval().to(DEVICE)


def load_gsddrunet():
    model = dinv.models.GSDRUNet(in_channels=1, out_channels=1, pretrained=None)
    sd = torch.load(WEIGHTS_DIR / "GSDRUNet_grayscale_torch.ckpt",
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model.eval().to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# Rician noise
# ═══════════════════════════════════════════════════════════════════════════════

def add_rician_noise_np(image: np.ndarray, sigma: float) -> np.ndarray:
    n1 = np.random.normal(0, sigma, image.shape)
    n2 = np.random.normal(0, sigma, image.shape)
    return np.sqrt((image + n1)**2 + n2**2)


def np2torch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)


def torch2np(x: torch.Tensor) -> np.ndarray:
    return x.squeeze().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# nmAPG  (GenRegBench protocol)
# ─────────────────────────────────────────────────────────────────────────────
# Minimise  E(x) = f(x) + lam * R(x)   via non-monotone APG
# f choices: L2 ||x-y||²/2,  L1 ||x-y||₁,  ElasticNet a||x-y||₁ + b||x-y||²/2
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _prox_l1(v: torch.Tensor, gamma: float) -> torch.Tensor:
    return v.sign() * (v.abs() - gamma).clamp(min=0)


def nmapg(y: torch.Tensor, grad_R, lam: float, fidelity: str = "l2",
          step: float = 0.1, n_iter: int = 300, tol: float = 1e-5,
          alpha: float = 1.0, beta_en: float = 0.5) -> torch.Tensor:
    """
    Non-monotone APG (Nesterov-like) for E(x) = data_fidelity(x,y) + lam*R(x).

    fidelity choices: 'l2' | 'l1' | 'elasticnet'
    For L1 / ElasticNet, the data term is handled via the proximal step (ADMM split).
    """
    x = y.clone()
    x_prev = x.clone()
    t = 1.0

    for k in range(n_iter):
        # ── gradient of data fidelity ──────────────────────────────────────
        if fidelity == "l2":
            grad_f = x - y                                # ∇||x-y||²/2
        elif fidelity == "l1":
            # Smooth L1 (Huber, δ=1e-3) for gradient-based update
            r = x - y
            grad_f = r / (r.abs() + 1e-3)
        else:  # elasticnet: α||x-y||₁ + β||x-y||²/2
            r = x - y
            grad_f = alpha * r / (r.abs() + 1e-3) + beta_en * r

        # ── full gradient ──────────────────────────────────────────────────
        with torch.no_grad():
            g = grad_f + lam * grad_R(x)

        x_new = (x - step * g).clamp(0, 1)

        # ── Nesterov momentum ──────────────────────────────────────────────
        t_new = (1 + (1 + 4*t**2)**0.5) / 2
        x_new = x_new + ((t-1)/t_new) * (x_new - x_prev)
        x_new = x_new.clamp(0, 1)

        if (x_new - x).abs().max().item() < tol:
            x = x_new; break

        x_prev, x, t = x.clone(), x_new, t_new

    return x.clamp(0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2-D solvers (pretrained)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_wcrr2d(y_np: np.ndarray, model: WCRR2D, lam: float = 0.05,
                 sigma: float = 0.1, fidelity: str = "l2") -> np.ndarray:
    y = np2torch(y_np)
    grad_R = lambda x: model.grad(x)
    out = nmapg(y, grad_R, lam=lam, fidelity=fidelity, step=0.1, n_iter=300)
    return torch2np(out)


def solve_gsddrunet2d(y_np: np.ndarray, model, sigma: float = 0.1,
                     fidelity: str = "l2", lam: float = 1.0,
                     tau: float = 0.5, n_iter: int = 30) -> np.ndarray:
    """PnP-GD with GSDRUNet denoiser and variable data fidelity."""
    y = np2torch(y_np)
    x = y.clone()
    sig_t = torch.tensor([[[[sigma]]]]).to(DEVICE)

    for _ in range(n_iter):
        if fidelity == "l2":
            grad_f = x - y
        elif fidelity == "l1":
            r = x - y
            grad_f = r / (r.abs() + 1e-3)
        else:
            r = x - y
            grad_f = lam * r / (r.abs() + 1e-3) + (1-lam) * r

        x_half = x - tau * grad_f
        with torch.no_grad():
            x = model(x_half, sigma).clamp(0, 1)

    return torch2np(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3-D solver (pretrained WCRR-3D)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_wcrr3d(vol_np: np.ndarray, model: WCRR3D, sigma: float,
                 lam: float = 0.05, fidelity: str = "l2",
                 n_iter: int = 100, step: float = 0.05) -> np.ndarray:
    """Run WCRR-3D on a full 3-D volume.  Input shape (D, H, W)."""
    # WCRR3D expects 2 input channels → tile
    y_t = torch.from_numpy(vol_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    y_t = y_t.expand(-1, 2, -1, -1, -1).clone()   # [1,2,D,H,W]
    sig_t = torch.tensor([sigma]).to(DEVICE)

    x = y_t.clone()
    x_prev = x.clone()
    t = 1.0

    for k in range(n_iter):
        if fidelity == "l2":
            grad_f = (x - y_t)
        elif fidelity == "l1":
            r = x - y_t
            grad_f = r / (r.abs() + 1e-3)
        else:
            r = x - y_t
            grad_f = 0.5 * r / (r.abs() + 1e-3) + 0.5 * r

        with torch.no_grad():
            g = grad_f + lam * model.grad(x, sig_t)

        x_new = (x - step * g).clamp(0, 1)
        t_new = (1 + (1 + 4*t**2)**0.5) / 2
        x_new = x_new + ((t-1)/t_new) * (x_new - x_prev)
        x_new = x_new.clamp(0, 1)

        if (x_new - x).abs().max().item() < 1e-5:
            x = x_new; break
        x_prev, x, t = x.clone(), x_new, t_new

    # Average over the 2 input channels, return single-channel
    return x[0].mean(0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def metrics(ref: np.ndarray, rec: np.ndarray) -> dict:
    with np.errstate(divide="ignore"):
        psnr = peak_signal_noise_ratio(ref, rec, data_range=1.0)
    ssim = structural_similarity(ref, rec, data_range=1.0, win_size=3)
    nrmse = np.linalg.norm(ref - rec) / (np.linalg.norm(ref) + 1e-8)
    return {"psnr": float(psnr if np.isfinite(psnr) else 999.0),
            "ssim": float(ssim), "nrmse": float(nrmse)}


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_panel(clean, noisy, results: dict, seq: str, sigma: float, out_dir: Path):
    n = len(results) + 2
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Clean"); axes[0].axis("off")
    axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Noisy sigma={sigma}"); axes[1].axis("off")
    for i, (name, img) in enumerate(results.items(), 2):
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(name, fontsize=7); axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{seq}_sigma{sigma:.2f}_panel.png", dpi=120, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading pretrained models...")
    wcrr2d = load_wcrr2d()
    gsddrunet = load_gsddrunet()
    wcrr3d = load_wcrr3d()
    print("  WCRR-2D OK   GSD-DRUNet OK   WCRR-3D OK")

    METHOD_2D = {
        "wcrr_l2":          lambda y, s: solve_wcrr2d(y, wcrr2d, lam=0.05, sigma=s, fidelity="l2"),
        "wcrr_l1":          lambda y, s: solve_wcrr2d(y, wcrr2d, lam=0.05, sigma=s, fidelity="l1"),
        "wcrr_elasticnet":  lambda y, s: solve_wcrr2d(y, wcrr2d, lam=0.05, sigma=s, fidelity="elasticnet"),
        "gsd_l2":           lambda y, s: solve_gsddrunet2d(y, gsddrunet, sigma=s, fidelity="l2"),
        "gsd_l1":           lambda y, s: solve_gsddrunet2d(y, gsddrunet, sigma=s, fidelity="l1"),
        "gsd_elasticnet":   lambda y, s: solve_gsddrunet2d(y, gsddrunet, sigma=s, fidelity="elasticnet"),
    }

    csv_path = OUT_DIR / "results_pretrained.csv"
    csv_exists = csv_path.exists()

    # Load already-completed rows to skip them on resume
    done_keys: set = set()
    if csv_exists:
        existing = pd.read_csv(csv_path)
        for _, r in existing.iterrows():
            done_keys.add((r["seq"], r["dim"], r["method"], float(r["sigma"])))
        print(f"Resuming: {len(done_keys)} rows already saved.")

    def _append_row(row: dict) -> None:
        key = (row["seq"], row["dim"], row["method"], float(row["sigma"]))
        if key in done_keys:
            return
        df_row = pd.DataFrame([row])
        df_row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done_keys.add(key)

    all_rows: list = []  # kept for final summary

    for seq, path, sl_idx in SEQUENCES:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq}")
        img_nib = nib.load(str(path))
        volume = img_nib.get_fdata().astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        slice_2d = volume[sl_idx]
        print(f"  2D slice: {slice_2d.shape}   3D volume: {volume.shape}")

        seq_dir = OUT_DIR / seq
        seq_dir.mkdir(exist_ok=True)

        # ── 2-D benchmark ─────────────────────────────────────────────────
        for sigma in NOISE_LEVELS:
            noisy = add_rician_noise_np(slice_2d, sigma)
            noisy_m = metrics(slice_2d, noisy)
            print(f"\n  sigma={sigma}  noisy PSNR={noisy_m['psnr']:.2f}")

            panel_imgs = {}
            for name, fn in METHOD_2D.items():
                key = (seq, "2D", name, float(sigma))
                if key in done_keys:
                    print(f"    {name:20s}  (skipped, already saved)")
                    continue
                t0 = time.time()
                denoised = fn(noisy, sigma)
                elapsed = time.time() - t0
                m = metrics(slice_2d, denoised)
                print(f"    {name:20s}  PSNR={m['psnr']:6.2f}  SSIM={m['ssim']:.4f}  NRMSE={m['nrmse']:.4f}  ({elapsed:.1f}s)")
                row = {"seq": seq, "dim": "2D", "method": name, "sigma": sigma, **m, "time_s": elapsed}
                _append_row(row)
                all_rows.append(row)
                panel_imgs[name] = denoised

            if panel_imgs:
                save_panel(slice_2d, noisy, panel_imgs, seq, sigma, seq_dir)

        # ── 3-D benchmark (WCRR-3D only, at sigma=0.1) ────────────────────────
        sigma = 0.1
        print(f"\n  3D WCRR (sigma={sigma}) ...")
        noisy_vol = add_rician_noise_np(volume, sigma)

        for fid in ("l2", "l1", "elasticnet"):
            name = f"wcrr3d_{fid}"
            key = (seq, "3D", name, float(sigma))
            if key in done_keys:
                print(f"    {name:20s}  (skipped, already saved)")
                continue
            t0 = time.time()
            denoised_vol = solve_wcrr3d(noisy_vol, wcrr3d, sigma=sigma,
                                        lam=0.05, fidelity=fid, n_iter=100)
            elapsed = time.time() - t0
            m = metrics(volume, denoised_vol)
            print(f"    {name:20s}  PSNR={m['psnr']:6.2f}  SSIM={m['ssim']:.4f}  NRMSE={m['nrmse']:.4f}  ({elapsed:.1f}s)")
            row = {"seq": seq, "dim": "3D", "method": name, "sigma": sigma, **m, "time_s": elapsed}
            _append_row(row)
            all_rows.append(row)

    # ── reload full CSV for summary ───────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"\nSaved {csv_path}")

    # ── summary table ─────────────────────────────────────────────────────
    print("\n── Mean PSNR (2D, all sigma) ─────────────────────────────")
    summary = (df[df.dim == "2D"]
               .groupby(["seq", "method"])["psnr"]
               .mean().round(2).unstack("method"))
    print(summary.to_string())

    print("\n── 3D WCRR PSNR (sigma=0.1) ──────────────────────────────")
    print(df[df.dim == "3D"][["seq","method","psnr","ssim"]].to_string(index=False))

    # ── save JSON summary ─────────────────────────────────────────────────
    summary_json = {
        "2d_mean_psnr": summary.to_dict(),
        "3d": df[df.dim=="3D"][["seq","method","psnr","ssim","nrmse"]].to_dict(orient="records"),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)


if __name__ == "__main__":
    main()
