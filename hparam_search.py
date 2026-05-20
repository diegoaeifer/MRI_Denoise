"""
3-hour hyperparameter search for WCRR-2D / WCRR-3D / GSD-DRUNet.
Varies: lam, step_size, n_iter, fidelity (L2/L1/ElasticNet/DynamicElasticNet).
Metrics: PSNR, SSIM, HaarPSI, EdgeLoss (Sobel MAE).
Results appended live to hparam_results.csv — safe to interrupt.
"""
from __future__ import annotations
import sys, os, time, json, random, itertools
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import torchcde
import nibabel as nib
import pandas as pd
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from piq import haarpsi
import deepinv as dinv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

WEIGHTS_DIR = Path(__file__).parent / "weights"
IXI_DIR     = Path("C:/projetos/Datasets/IXI/all")
OUT_DIR     = Path("C:/projetos/benchmark_results/hparam")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH    = OUT_DIR / "hparam_results.csv"

SEARCH_DURATION_S = 3 * 3600  # 3 hours

# Use T2 (good SNR, moderate complexity) as the reference sequence for 2D search
# Use PD slice for 3D search (smaller than MRA)
SLICE_2D_PATH  = IXI_DIR / "IXI015-HH-1258-T2.nii.gz"
SLICE_2D_IDX   = 128
VOL_3D_PATH    = IXI_DIR / "IXI015-HH-1258-PD.nii.gz"

NOISE_LEVELS   = [0.05, 0.1, 0.2, 0.3]

# ─── model definitions (same as benchmark_pretrained.py) ──────────────────────

class ZeroMean(nn.Module):
    def forward(self, x): return x - x.mean(dim=(1,2,3), keepdim=True)

class WCRR2D(dinv.optim.Prior):
    def __init__(self, sigma=0.1, weak_convexity=1.0,
                 nb_channels=(1,4,8,64), filter_sizes=(5,5,5)):
        super().__init__()
        self.nb_filters = nb_channels[-1]; self.in_channels = nb_channels[0]
        fsize = sum(filter_sizes) - len(filter_sizes) + 1
        self.filters = nn.Sequential(*[
            nn.Conv2d(nb_channels[i], nb_channels[i+1], filter_sizes[i],
                      padding=filter_sizes[i]//2, bias=False)
            for i in range(len(filter_sizes))])
        P.register_parametrization(self.filters[0], "weight", ZeroMean())
        self.dirac = torch.zeros(nb_channels[0], nb_channels[0], 2*fsize-1, 2*fsize-1)
        self.dirac[:, :, fsize-1, fsize-1] = 1.0
        self.scaling = nn.Parameter(
            torch.log(torch.tensor(2.0)/sigma) * torch.ones(1, self.nb_filters, 1, 1))
        self.beta = nn.Parameter(torch.tensor(4.0)); self.weak_cvx = weak_convexity
    def _grad_sl1(self, x): return torch.clip(x, -1.0, 1.0)
    def _conv_lip(self):
        imp = self.filters(self.dirac.to(next(self.parameters()).device))
        for f in reversed(self.filters): imp = F.conv_transpose2d(imp, f.weight, padding=f.padding)
        return torch.fft.fft2(imp, s=[256,256]).abs().max()
    def _conv(self, x): return self.filters(x / self._conv_lip().sqrt())
    def _conv_T(self, x):
        x = x / self._conv_lip().sqrt()
        for f in reversed(self.filters): x = F.conv_transpose2d(x, f.weight, padding=f.padding)
        return x
    def grad(self, x, *a, **k):
        exp = x.shape[1]==1 and self.in_channels>1
        if exp: x = x.tile(1, self.in_channels, 1, 1)
        g = self._conv(x) * torch.exp(self.scaling)
        g = self._grad_sl1(torch.exp(self.beta)*g) - self._grad_sl1(g)*self.weak_cvx
        g = self._conv_T(g * torch.exp(-self.scaling))
        return g.sum(1, keepdim=True) if exp else g
    def _apply(self, fn): self.dirac = fn(self.dirac); return super()._apply(fn)

class LinearSpline(nn.Module):
    def __init__(self, N=32, K=12, sigma_min=0.01, sigma_max=0.1, eps=1e-5):
        super().__init__(); self.N=N; self.K=K; self.eps=float(eps)
        self.register_buffer("t_knots", torch.linspace(sigma_min, sigma_max, K))
        self.s_at_knots = nn.Parameter(torch.zeros(1, K, N))
    def forward(self, sigma):
        sigma = sigma.view(-1)
        coeffs = torchcde.linear_interpolation_coeffs(self.s_at_knots, t=self.t_knots)
        s_vals = torchcde.LinearInterpolation(coeffs).evaluate(sigma).squeeze(0)
        return (torch.exp(s_vals) / (sigma.view(-1,1) + self.eps)).view(len(sigma), self.N, 1,1,1)

class ZeroMean3D(nn.Module):
    def forward(self, x): return x - x.mean(dim=(1,2,3,4), keepdim=True)

class WCRR3D(dinv.optim.Prior):
    def __init__(self, weak_convexity=1.0, nb_channels=(2,4,8,32),
                 filter_sizes=(3,3,3), rotations=True):
        super().__init__()
        self.nb_filters=nb_channels[-1]; self.weak_cvx=weak_convexity; self.rotations=rotations
        fsize = sum(filter_sizes) - len(filter_sizes) + 1
        self.filters = nn.Sequential(*[
            nn.Conv3d(nb_channels[i], nb_channels[i+1], filter_sizes[i],
                      padding=filter_sizes[i]//2, bias=False)
            for i in range(len(filter_sizes))])
        P.register_parametrization(self.filters[0], "weight", ZeroMean3D())
        sz = 2*fsize-1; in_c = self.filters[0].in_channels
        self.dirac = torch.zeros(1, in_c, sz, sz, sz)
        self.dirac[0, :, fsize-1, fsize-1, fsize-1] = 1.0
        self.scaling = LinearSpline(N=self.nb_filters, K=12)
        self.beta = nn.Parameter(torch.tensor(4.0))
    def _grad_sl1(self, x): return torch.clip(x, -1.0, 1.0)
    def _conv_lip(self):
        imp = self.filters(self.dirac.to(next(self.parameters()).device))
        for f in reversed(self.filters): imp = F.conv_transpose3d(imp, f.weight, padding=f.padding)
        return torch.fft.fftn(imp.float()).abs().max().to(imp.dtype)
    def _conv(self, x): return self.filters(x / self._conv_lip().sqrt())
    def _conv_T(self, x):
        x = x / self._conv_lip().sqrt()
        for f in reversed(self.filters): x = F.conv_transpose3d(x, f.weight, padding=f.padding)
        return x
    def _grad_R(self, x, sc):
        b = torch.exp(self.beta); g = self._conv(x)*sc
        return self._conv_T((self._grad_sl1(b*g) - self._grad_sl1(g)*self.weak_cvx) / sc)
    def grad(self, x, sigma):
        sc = self.scaling(sigma)
        if self.rotations:
            xDH=torch.rot90(x,1,(-3,-2)); xDW=torch.rot90(x,1,(-3,-1)); xHW=torch.rot90(x,1,(-2,-1))
            return (self._grad_R(x,sc)
                    + torch.rot90(self._grad_R(xDH,sc),-1,(-3,-2))
                    + torch.rot90(self._grad_R(xDW,sc),-1,(-3,-1))
                    + torch.rot90(self._grad_R(xHW,sc),-1,(-2,-1))) / 4
        return self._grad_R(x, sc)
    def _apply(self, fn): self.dirac=fn(self.dirac); return super()._apply(fn)

# ─── load models ──────────────────────────────────────────────────────────────

def _load(model, path):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model.eval().to(DEVICE)

print("Loading models...")
WCRR2D_MODEL   = _load(WCRR2D(sigma=0.1, weak_convexity=1.0, nb_channels=(1,4,8,64)), WEIGHTS_DIR/"WCRR_gray.pt")
WCRR3D_MODEL   = _load(WCRR3D(weak_convexity=1.0, nb_channels=(2,4,8,32), filter_sizes=(3,3,3)), WEIGHTS_DIR/"WCRR_3d.pt")
GSD_MODEL      = _load(dinv.models.GSDRUNet(in_channels=1, out_channels=1, pretrained=None), WEIGHTS_DIR/"GSDRUNet_grayscale_torch.ckpt")
print("Models loaded.")

# ─── noise ────────────────────────────────────────────────────────────────────

def add_rician(x, sigma):
    n1 = np.random.normal(0, sigma, x.shape); n2 = np.random.normal(0, sigma, x.shape)
    return np.sqrt((x+n1)**2 + n2**2)

def np2t(x): return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
def t2np(x): return x.squeeze().cpu().numpy()

# ─── metrics ──────────────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
_SOBEL_Y = _SOBEL_X.transpose(2,3)

def edge_loss(ref, rec):
    """Mean absolute Sobel-gradient error (edge preservation)."""
    sx = _SOBEL_X.to(DEVICE); sy = _SOBEL_Y.to(DEVICE)
    ref_t = np2t(ref); rec_t = np2t(rec)
    er = (F.conv2d(ref_t, sx, padding=1)**2 + F.conv2d(ref_t, sy, padding=1)**2).sqrt()
    ec = (F.conv2d(rec_t, sx, padding=1)**2 + F.conv2d(rec_t, sy, padding=1)**2).sqrt()
    return (er - ec).abs().mean().item()

def compute_all_metrics(ref: np.ndarray, rec: np.ndarray) -> dict:
    with np.errstate(divide="ignore"):
        psnr_val = peak_signal_noise_ratio(ref, rec, data_range=1.0)
    ssim_val = structural_similarity(ref, rec, data_range=1.0, win_size=3)
    nrmse    = np.linalg.norm(ref-rec) / (np.linalg.norm(ref) + 1e-8)
    # HaarPSI expects [B,C,H,W] — use middle 2D slice for 3D arrays
    if ref.ndim == 3:
        mid = ref.shape[0] // 2
        r2 = ref[mid].astype(np.float32); c2 = rec[mid].astype(np.float32)
    else:
        r2 = ref.astype(np.float32); c2 = rec.astype(np.float32)
    r = np2t(r2); c = np2t(c2)
    hp = haarpsi(r, c, data_range=1.0).item()
    el = edge_loss(r2, c2)
    return {"psnr": float(psnr_val if np.isfinite(psnr_val) else 999.),
            "ssim": float(ssim_val), "nrmse": float(nrmse),
            "haarpsi": float(hp), "edge_loss": float(el)}

# ─── data-fidelity gradient helpers ───────────────────────────────────────────

def df_grad(x, y, fidelity: str, alpha=0.5, beta=0.5, delta=1e-2, k=None, n_iter=None):
    """Gradient of data fidelity term w.r.t. x."""
    r = x - y
    if fidelity == "l2":
        return r
    elif fidelity == "l1":
        return r / (r.abs() + delta)
    elif fidelity == "elasticnet":
        return alpha * r / (r.abs() + delta) + beta * r
    else:  # dynamic elasticnet: alpha anneals 1->0, beta grows 0->1 over iterations
        t = (k / n_iter) if (k is not None and n_iter is not None) else 0.5
        a = alpha * (1 - t); b = beta * t
        return a * r / (r.abs() + delta) + b * r

# ─── nmAPG solver ─────────────────────────────────────────────────────────────

@torch.no_grad()
def nmapg(y_t, grad_R_fn, lam, fidelity, step, n_iter, tol=1e-5,
          alpha=0.5, beta=0.5, delta=1e-2):
    x = y_t.clone(); x_prev = x.clone(); t = 1.0
    for k in range(n_iter):
        g = df_grad(x, y_t, fidelity, alpha, beta, delta, k, n_iter) + lam * grad_R_fn(x)
        xn = (x - step * g).clamp(0, 1)
        tn = (1 + (1 + 4*t**2)**0.5) / 2
        xn = (xn + ((t-1)/tn) * (xn - x_prev)).clamp(0, 1)
        if (xn - x).abs().max().item() < tol: x = xn; break
        x_prev, x, t = x.clone(), xn, tn
    return x.clamp(0, 1)

# ─── 2D WCRR solver ───────────────────────────────────────────────────────────

def run_wcrr2d(noisy_np, cfg):
    y = np2t(noisy_np)
    out = nmapg(y, WCRR2D_MODEL.grad, cfg["lam"], cfg["fidelity"],
                cfg["step"], cfg["n_iter"],
                alpha=cfg.get("alpha", 0.5), beta=cfg.get("beta_en", 0.5),
                delta=cfg.get("delta", 1e-2))
    return t2np(out)

# ─── GSD-DRUNet PnP solver ────────────────────────────────────────────────────

@torch.no_grad()
def run_gsd(noisy_np, sigma_noise, cfg):
    y = np2t(noisy_np); x = y.clone()
    fid = cfg["fidelity"]; tau = cfg["tau"]; n_iter = cfg["n_iter"]
    for k in range(n_iter):
        gf = df_grad(x, y, fid, cfg.get("alpha",0.5), cfg.get("beta_en",0.5),
                     cfg.get("delta",1e-2), k, n_iter)
        x = GSD_MODEL((x - tau*gf).clamp(0,1), sigma_noise).clamp(0,1)
    return t2np(x)

# ─── 3D WCRR solver ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_wcrr3d(vol_np, sigma, cfg):
    y = torch.from_numpy(vol_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    y = y.expand(-1, 2, -1, -1, -1).clone()
    sig_t = torch.tensor([sigma]).to(DEVICE)
    x = y.clone(); x_prev = x.clone(); t = 1.0
    for k in range(cfg["n_iter"]):
        g = df_grad(x, y, cfg["fidelity"], cfg.get("alpha",0.5), cfg.get("beta_en",0.5),
                    cfg.get("delta",1e-2), k, cfg["n_iter"]) + cfg["lam"] * WCRR3D_MODEL.grad(x, sig_t)
        xn = (x - cfg["step"]*g).clamp(0,1)
        tn = (1 + (1+4*t**2)**0.5)/2
        xn = (xn + ((t-1)/tn)*(xn-x_prev)).clamp(0,1)
        if (xn-x).abs().max().item() < 1e-5: x=xn; break
        x_prev, x, t = x.clone(), xn, tn
    return x[0].mean(0).cpu().numpy()

# ─── search spaces ────────────────────────────────────────────────────────────

FIDELITIES = ["l2", "l1", "elasticnet", "dynamic_elasticnet"]

WCRR2D_GRID = {
    "lam":      [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "step":     [0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
    "n_iter":   [100, 200, 300, 500],
    "fidelity": FIDELITIES,
    "alpha":    [0.3, 0.5, 0.7],
    "beta_en":  [0.3, 0.5, 0.7],
    "delta":    [1e-3, 5e-3, 1e-2, 5e-2],
}

GSD_GRID = {
    "tau":      [0.1, 0.2, 0.3, 0.5, 0.7],
    "n_iter":   [20, 30, 50, 80],
    "fidelity": FIDELITIES,
    "alpha":    [0.3, 0.5, 0.7],
    "beta_en":  [0.3, 0.5, 0.7],
    "delta":    [1e-3, 5e-3, 1e-2, 5e-2],
}

WCRR3D_GRID = {
    "lam":      [0.01, 0.03, 0.05, 0.1, 0.15],
    "step":     [0.02, 0.05, 0.08, 0.1],
    "n_iter":   [50, 80, 100],
    "fidelity": FIDELITIES,
    "alpha":    [0.3, 0.5, 0.7],
    "beta_en":  [0.3, 0.5, 0.7],
    "delta":    [1e-3, 5e-3, 1e-2],
}

# ─── load reference data ──────────────────────────────────────────────────────

def load_ref_2d():
    img = nib.load(str(SLICE_2D_PATH)).get_fdata().astype(np.float32)
    sl  = img[SLICE_2D_IDX]
    return (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)

def load_ref_3d():
    vol = nib.load(str(VOL_3D_PATH)).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    d, h, w = vol.shape
    # clamp indices to valid range
    d0 = max(0, d//2-32);  d1 = min(d, d//2+32)
    h0 = max(0, h//2-64);  h1 = min(h, h//2+64)
    w0 = max(0, w//2-64);  w1 = min(w, w//2+64)
    return vol[d0:d1, h0:h1, w0:w1]

# ─── random config sampler ────────────────────────────────────────────────────

def sample_cfg(grid: dict) -> dict:
    return {k: random.choice(v) for k, v in grid.items()}

# ─── CSV helpers ──────────────────────────────────────────────────────────────

_COLUMNS = ["model", "seq_type", "sigma", "fidelity", "lam", "step", "n_iter",
            "tau", "alpha", "beta_en", "delta",
            "psnr", "ssim", "nrmse", "haarpsi", "edge_loss", "time_s"]

def append_row(row: dict):
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in _COLUMNS}])
    df.to_csv(CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)

# ─── best-config tracker ──────────────────────────────────────────────────────

best: dict = {}  # model -> {metric -> (value, cfg)}

def update_best(model, cfg, m):
    if model not in best: best[model] = {}
    for metric in ("psnr", "ssim", "haarpsi"):
        higher_is_better = True
        val = m[metric]
        if metric not in best[model] or val > best[model][metric][0]:
            best[model][metric] = (val, dict(cfg))
    for metric in ("nrmse", "edge_loss"):
        val = m[metric]
        if metric not in best[model] or val < best[model][metric][0]:
            best[model][metric] = (val, dict(cfg))

def print_best():
    print("\n--- Current best configs ---")
    for model, metrics_d in best.items():
        print(f"  {model}:")
        for metric, (val, cfg) in metrics_d.items():
            print(f"    {metric}: {val:.4f}  cfg={cfg}")

# ─── main search loop ─────────────────────────────────────────────────────────

def main():
    ref_2d = load_ref_2d()
    ref_3d = load_ref_3d()
    print(f"Ref 2D shape: {ref_2d.shape}   Ref 3D shape: {ref_3d.shape}")

    np.random.seed(42); random.seed(42)

    t_start = time.time()
    itr = 0
    eval_sigma = 0.1  # fix sigma to keep search fast; covers mid-range noise

    print(f"\nStarting {SEARCH_DURATION_S//3600}h hyperparameter search...")
    print(f"Results -> {CSV_PATH}\n")

    while time.time() - t_start < SEARCH_DURATION_S:
        elapsed = time.time() - t_start
        remaining = SEARCH_DURATION_S - elapsed
        itr += 1

        # Rotate through: wcrr2d, gsd, wcrr3d (3D is slower, do less often)
        model_choice = ["wcrr2d", "gsd", "wcrr2d", "gsd", "wcrr3d"][itr % 5]

        try:
            if model_choice == "wcrr2d":
                cfg = sample_cfg(WCRR2D_GRID)
                noisy = add_rician(ref_2d, eval_sigma)
                t0 = time.time()
                denoised = run_wcrr2d(noisy, cfg)
                elapsed_run = time.time() - t0
                m = compute_all_metrics(ref_2d, denoised)
                update_best("wcrr2d", cfg, m)
                row = {"model": "wcrr2d", "seq_type": "T2", "sigma": eval_sigma,
                       **cfg, "tau": np.nan, **m, "time_s": elapsed_run}

            elif model_choice == "gsd":
                cfg = sample_cfg(GSD_GRID)
                noisy = add_rician(ref_2d, eval_sigma)
                t0 = time.time()
                denoised = run_gsd(noisy, eval_sigma, cfg)
                elapsed_run = time.time() - t0
                m = compute_all_metrics(ref_2d, denoised)
                update_best("gsd", cfg, m)
                row = {"model": "gsd", "seq_type": "T2", "sigma": eval_sigma,
                       **cfg, "lam": np.nan, "step": np.nan, **m, "time_s": elapsed_run}

            else:  # wcrr3d
                cfg = sample_cfg(WCRR3D_GRID)
                noisy_vol = add_rician(ref_3d, eval_sigma)
                t0 = time.time()
                denoised_vol = run_wcrr3d(noisy_vol, eval_sigma, cfg)
                elapsed_run = time.time() - t0
                m = compute_all_metrics(ref_3d, denoised_vol)
                update_best("wcrr3d", cfg, m)
                row = {"model": "wcrr3d", "seq_type": "PD_3D", "sigma": eval_sigma,
                       **cfg, "tau": np.nan, **m, "time_s": elapsed_run}

            append_row(row)
            print(f"[{elapsed/60:.1f}min / itr={itr}] {model_choice:8s} "
                  f"fid={cfg['fidelity']:18s}  "
                  f"PSNR={m['psnr']:6.2f}  SSIM={m['ssim']:.4f}  "
                  f"HaarPSI={m['haarpsi']:.4f}  EdgeLoss={m['edge_loss']:.4f}  ({elapsed_run:.1f}s)",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{elapsed/60:.1f}min / itr={itr}] {model_choice:8s}  CUDA OOM — skipped, cache cleared", flush=True)
            continue
        except Exception as exc:
            print(f"[{elapsed/60:.1f}min / itr={itr}] {model_choice:8s}  ERROR: {exc} — skipped", flush=True)
            continue
        finally:
            torch.cuda.empty_cache()

        if itr % 20 == 0:
            print_best()

    print("\n=== Search complete ===")
    print_best()

    # Save best configs to JSON
    best_json = {}
    for model, metrics_d in best.items():
        best_json[model] = {metric: {"value": float(v), "config": c}
                            for metric, (v, c) in metrics_d.items()}
    with open(OUT_DIR / "best_configs.json", "w") as f:
        json.dump(best_json, f, indent=2)
    print(f"Best configs saved to {OUT_DIR/'best_configs.json'}")

    # Final summary table from CSV
    df = pd.read_csv(CSV_PATH)
    print("\nTop 5 per model by PSNR:")
    print(df.groupby("model").apply(lambda g: g.nlargest(5, "psnr")[
        ["model","fidelity","lam","step","n_iter","tau","psnr","ssim","haarpsi","edge_loss"]
    ]).to_string())

if __name__ == "__main__":
    main()
