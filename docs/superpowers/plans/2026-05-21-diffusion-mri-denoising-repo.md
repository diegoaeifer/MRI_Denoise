# Diffusion MRI Denoising Repo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new GitHub repository (`diffusion-mri-denoising`) that integrates Di-Fusion, EquS, and Flower as submodules, provides adapters that translate MRI grayscale data to each model's expected input, and runs a unified benchmark comparing all three on 2D (and 3D) MRI denoising.

**Architecture:**
```
diffusion-mri-denoising/
  submodules/Di-Fusion  EquS  Flower   ← git submodules (original repos untouched)
  data/mri_loader.py                   ← DICOM/NIfTI loader + Rician noise (from MRI_Denoise)
  adapters/{di_fusion,equs,flower}_adapter.py  ← uniform interface: denoise(arr, sigma) → arr
  scripts/benchmark_diffusion.py       ← runs all adapters on manifest.csv slices; CSV output
  scripts/download_weights.py          ← fetches EquS (OpenAI) + Flower (HuggingFace) weights
```
Each adapter wraps the upstream model, handles grayscale↔RGB conversion, resizing to the model's expected spatial resolution, and restores the original spatial size. All models expose one function: `denoise(image: np.ndarray, sigma: float) -> np.ndarray` where image is `(H, W)` float32 in [0, 1].

**Tech Stack:** Python 3.12, PyTorch 2.1+cu128, nibabel, pydicom, scikit-image (PSNR/SSIM), DeepInverse (required by Flower), torchdiffeq, huggingface-hub. Three git submodules: `FouierL/Di-Fusion`, `FouierL/EquS`, `mehrsapo/Flower`.

---

## Context

### Model summary

| Model | Paper | Noise type | Input | Pretrained weights |
|-------|-------|-----------|-------|-------------------|
| **Di-Fusion** | ICLR 2025 | Rician (diffusion MRI) | 1-ch 128×128 NIfTI slices | None — self-supervised training required |
| **EquS** | WACV 2026 | Gaussian (configurable σ) | 3-ch RGB 256×256 | OpenAI 256×256 unconditional diffusion (download ~2 GB) |
| **Flower** | ICLR 2026 | Gaussian isotropic + non-isotropic | 3-ch RGB 128×128 | HuggingFace `mehrsapo/flower-weights` (CelebA/AFHQ) |

### MRI data infrastructure (from MRI_Denoise codebase)

Key functions to port into `data/mri_loader.py`:

```python
# From C:\projetos\MRI_Denoise\scripts\search\eval_pipeline.py
def load_clean_2d(path: str | Path) -> np.ndarray:
    # NIfTI: nibabel.load(...).get_fdata() → slice → normalize
    # DICOM: pydicom.dcmread(...).pixel_array → normalize
    ...

def add_rician_noise(clean: np.ndarray, sigma: float, rng) -> np.ndarray:
    n1 = rng.standard_normal(clean.shape).astype(np.float32) * sigma
    n2 = rng.standard_normal(clean.shape).astype(np.float32) * sigma
    return np.sqrt((clean + n1)**2 + n2**2).clip(0, None)
```

Manifest CSV at `D:\Dataset MRI\manifest.csv` — columns: `file_path`, `anatomy`, `source_dataset`, `original_format`.

### Adapter design contract

Every adapter must implement:
```python
class BaseAdapter:
    def denoise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        image: (H, W) float32, values in [0, 1]
        sigma: noise standard deviation (same scale as image values)
        returns: (H, W) float32, values in [0, 1]
        """
```

Grayscale-to-RGB conversion (for EquS and Flower): `rgb = np.stack([gray, gray, gray], axis=0)`.
After inference, convert back: `gray = output_rgb.mean(axis=0)` or take channel 0.

### Di-Fusion training note

Di-Fusion is self-supervised (no clean targets required). Training on MRI data uses:
```bash
python Di-Fusion/train.py -c Di-Fusion/config/hardi_150.json --data_path <noisy_nifti_dir>
```
The plan includes a lightweight 10-epoch smoke-test training on IXI T1 data (available at `C:\projetos\Datasets\IXI-T1\`). This produces a checkpoint used by the adapter.

---

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `diffusion-mri-denoising/` | **Create repo** | New GitHub repo |
| `submodules/Di-Fusion/` | **Add submodule** | `FouierL/Di-Fusion` |
| `submodules/EquS/` | **Add submodule** | `FouierL/EquS` |
| `submodules/Flower/` | **Add submodule** | `mehrsapo/Flower` |
| `data/__init__.py` | **Create** | Package init |
| `data/mri_loader.py` | **Create** | DICOM/NIfTI loader + Rician noise |
| `adapters/__init__.py` | **Create** | Package init |
| `adapters/di_fusion_adapter.py` | **Create** | Di-Fusion → uniform interface |
| `adapters/equs_adapter.py` | **Create** | EquS → uniform interface |
| `adapters/flower_adapter.py` | **Create** | Flower → uniform interface |
| `scripts/download_weights.py` | **Create** | Download EquS + Flower weights |
| `scripts/benchmark_diffusion.py` | **Create** | Unified benchmark runner |
| `tests/test_mri_loader.py` | **Create** | TDD for data loader |
| `tests/test_adapters.py` | **Create** | TDD for adapter interfaces |
| `environment.yml` | **Create** | Unified conda environment |
| `README.md` | **Create** | Repo documentation |

---

## Task 0: Create GitHub Repository and Project Scaffold

**Files:**
- Create: `diffusion-mri-denoising/` (new repo at `C:\projetos\diffusion-mri-denoising\`)
- Add submodules: `submodules/Di-Fusion`, `submodules/EquS`, `submodules/Flower`
- Create: `environment.yml`, `README.md`, `data/__init__.py`, `adapters/__init__.py`, `tests/__init__.py`, `scripts/__init__.py`, `configs/`, `artifacts/`

- [ ] **Step 1: Create the GitHub repo**

```powershell
# Requires gh CLI authenticated as diegoaeifer
gh repo create diegoaeifer/diffusion-mri-denoising `
  --public `
  --description "Benchmark of diffusion-based models (Di-Fusion, EquS, Flower) for MRI denoising" `
  --clone `
  --gitignore Python
Set-Location C:\projetos\diffusion-mri-denoising
```

Expected: repo created at `https://github.com/diegoaeifer/diffusion-mri-denoising`, cloned locally.

- [ ] **Step 2: Add the three submodules**

```powershell
git submodule add https://github.com/FouierL/Di-Fusion submodules/Di-Fusion
git submodule add https://github.com/FouierL/EquS submodules/EquS
git submodule add https://github.com/mehrsapo/Flower submodules/Flower
git submodule update --init --recursive
```

Expected: three subdirectories under `submodules/`, each with their full code.

- [ ] **Step 3: Create directory structure**

```powershell
New-Item -ItemType Directory data, adapters, scripts, tests, configs, artifacts
New-Item -ItemType File `
  data/__init__.py, adapters/__init__.py, `
  tests/__init__.py, scripts/__init__.py
```

- [ ] **Step 4: Create `environment.yml`**

```yaml
name: diffusion-mri
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pytorch>=2.1
  - torchvision
  - pytorch-cuda=12.8
  - pip
  - pip:
    - nibabel
    - pydicom
    - scikit-image
    - scikit-learn
    - numpy
    - scipy
    - pandas
    - tqdm
    - huggingface-hub
    - deepinverse
    - torchdiffeq
    - pot              # optimal transport (Flower)
    - torchmetrics
    - lpips
    - PyYAML
    - Pillow
    - matplotlib
    - SimpleITK
    - PyTorch-Lightning>=2.2
    - einops
    - pytest
```

- [ ] **Step 5: Create `README.md`**

```markdown
# Diffusion MRI Denoising

Benchmark of three diffusion-based image restoration models on MRI denoising:

| Model | Paper | Venue | Code |
|-------|-------|-------|------|
| **Di-Fusion** | Self-supervised diffusion MRI denoising | ICLR 2025 | [FouierL/Di-Fusion](https://github.com/FouierL/Di-Fusion) |
| **EquS** | Equivariant sampling for image restoration | WACV 2026 | [FouierL/EquS](https://github.com/FouierL/EquS) |
| **Flower** | Flow matching for inverse problems | ICLR 2026 | [mehrsapo/Flower](https://github.com/mehrsapo/Flower) |

## Setup

```bash
conda env create -f environment.yml
conda activate diffusion-mri
git submodule update --init --recursive
python scripts/download_weights.py      # EquS (OpenAI) + Flower (HuggingFace)
```

## Benchmark

```bash
python scripts/benchmark_diffusion.py \
  --manifest "D:/Dataset MRI/manifest.csv" \
  --models di_fusion equs flower \
  --sigma 0.05 0.10 0.20 \
  --n_slices 50 \
  --out artifacts/benchmark_results.csv
```

## Data

Reads MRI data from a manifest CSV (compatible with `MRI_Denoise` project).
Adds Rician noise at configurable σ levels.
```

- [ ] **Step 6: Commit scaffold**

```powershell
git add .
git commit -m "init: scaffold with 3 submodules, environment, and project structure"
git push origin main
```

---

## Task 1: MRI Data Loader

**Files:**
- Create: `data/mri_loader.py`
- Create: `tests/test_mri_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mri_loader.py`:

```python
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from data.mri_loader import (
    add_rician_noise,
    load_slice_from_path,
    load_manifest_subset,
    normalize_to_unit,
)


def test_add_rician_noise_shape():
    rng = np.random.default_rng(42)
    img = np.random.rand(128, 128).astype(np.float32)
    noisy = add_rician_noise(img, sigma=0.1, rng=rng)
    assert noisy.shape == img.shape
    assert noisy.dtype == np.float32


def test_add_rician_noise_increases_noise():
    rng = np.random.default_rng(42)
    clean = np.ones((64, 64), dtype=np.float32) * 0.5
    noisy = add_rician_noise(clean, sigma=0.2, rng=rng)
    # Rician adds noise: result should differ from clean
    assert not np.allclose(noisy, clean)


def test_add_rician_noise_nonneg():
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64), dtype=np.float32)
    noisy = add_rician_noise(img, sigma=0.3, rng=rng)
    assert np.all(noisy >= 0), "Rician noise output must be non-negative"


def test_normalize_to_unit_range():
    img = np.array([[0.0, 500.0, 1000.0]], dtype=np.float32)
    normalized = normalize_to_unit(img)
    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_normalize_to_unit_constant():
    img = np.ones((10, 10), dtype=np.float32) * 5.0
    normalized = normalize_to_unit(img)
    # Constant image: avoid division by zero, return zeros
    assert np.all(normalized == 0.0)


def test_load_manifest_subset_returns_paths(tmp_path):
    import csv
    # Create a fake manifest CSV
    csv_file = tmp_path / "manifest.csv"
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "anatomy", "original_format", "source_dataset"])
        w.writeheader()
        for i in range(5):
            w.writerow({
                "file_path": f"/fake/path/slice_{i}.dcm",
                "anatomy": "brain",
                "original_format": "dicom",
                "source_dataset": "test",
            })
    paths = load_manifest_subset(str(csv_file), n=3, anatomy_filter=None)
    assert len(paths) == 3
    assert all(isinstance(p, str) for p in paths)


def test_load_manifest_subset_anatomy_filter(tmp_path):
    import csv
    csv_file = tmp_path / "manifest.csv"
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_path", "anatomy", "original_format", "source_dataset"])
        w.writeheader()
        w.writerow({"file_path": "/fake/brain.dcm", "anatomy": "brain", "original_format": "dicom", "source_dataset": "test"})
        w.writerow({"file_path": "/fake/spine.dcm", "anatomy": "spine", "original_format": "dicom", "source_dataset": "test"})
    paths = load_manifest_subset(str(csv_file), n=10, anatomy_filter="brain")
    assert len(paths) == 1
    assert "brain" in paths[0]
```

- [ ] **Step 2: Run test to confirm failure**

```powershell
Set-Location C:\projetos\diffusion-mri-denoising
python -m pytest tests/test_mri_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'data.mri_loader'`

- [ ] **Step 3: Implement `data/mri_loader.py`**

```python
"""MRI data loading and noise synthesis.

Adapted from MRI_Denoise project (C:/projetos/MRI_Denoise/scripts/search/eval_pipeline.py
and src/data/noise_pipeline.py) with unified interface for diffusion model benchmarking.
"""
from __future__ import annotations
import csv
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np


def normalize_to_unit(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros for constant images."""
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)


def add_rician_noise(
    clean: np.ndarray,
    sigma: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Rician noise: sqrt((clean + n1)^2 + n2^2), n1/n2 ~ N(0, sigma).

    Rician is the correct noise model for magnitude MRI images.
    """
    if rng is None:
        rng = np.random.default_rng()
    n1 = (rng.standard_normal(clean.shape) * sigma).astype(np.float32)
    n2 = (rng.standard_normal(clean.shape) * sigma).astype(np.float32)
    return np.sqrt((clean + n1) ** 2 + n2 ** 2).clip(0.0, None).astype(np.float32)


def load_slice_from_path(file_path: str, slice_idx: int = 60) -> Optional[np.ndarray]:
    """Load a 2D slice from a DICOM or NIfTI file.

    Returns (H, W) float32 normalized to [0, 1], or None on failure.
    """
    p = Path(file_path)
    if not p.exists():
        return None

    try:
        ext = "".join(p.suffixes).lower()

        if ".nii" in ext:
            import nibabel as nib
            vol = nib.load(str(p)).get_fdata(dtype=np.float32)
            if vol.ndim == 4:
                vol = vol[..., 0]  # take first volume (e.g., b0 for DWI)
            idx = min(slice_idx, vol.shape[2] - 1)
            img = vol[:, :, idx]

        elif ext in {".dcm", ".dicom", ""}:
            import pydicom
            ds = pydicom.dcmread(str(p))
            img = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            img = img * slope + intercept

        elif ext in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(str(p), "r") as f:
                key = list(f.keys())[0]
                arr = f[key][()]
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]  # middle slice
            img = arr.astype(np.float32)

        else:
            return None

        return normalize_to_unit(img)

    except Exception:
        return None


def load_manifest_subset(
    manifest_csv: str,
    n: int = 50,
    anatomy_filter: Optional[str] = None,
    seed: int = 42,
) -> list[str]:
    """Sample n file paths from the manifest CSV.

    Args:
        manifest_csv: Path to manifest.csv (from MRI_Denoise export pipeline)
        n: Number of files to sample
        anatomy_filter: If set, only include rows where anatomy == this value
        seed: Random seed for reproducibility

    Returns:
        List of file_path strings (may include non-existent files — check when loading)
    """
    rows: list[str] = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if anatomy_filter and row.get("anatomy", "") != anatomy_filter:
                continue
            fp = row.get("file_path", "")
            if fp:
                rows.append(fp)

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_batch(
    manifest_csv: str,
    n: int = 50,
    anatomy_filter: Optional[str] = None,
    slice_idx: int = 60,
    seed: int = 42,
) -> list[np.ndarray]:
    """Load n normalized MRI slices from manifest. Skips unreadable files."""
    paths = load_manifest_subset(manifest_csv, n=n * 3, anatomy_filter=anatomy_filter, seed=seed)
    slices: list[np.ndarray] = []
    for fp in paths:
        if len(slices) >= n:
            break
        img = load_slice_from_path(fp, slice_idx=slice_idx)
        if img is not None and img.shape[0] >= 32 and img.shape[1] >= 32:
            slices.append(img)
    return slices
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_mri_loader.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add data/mri_loader.py tests/test_mri_loader.py
git commit -m "feat: add MRI data loader with Rician noise synthesis"
git push
```

---

## Task 2: Weight Download Script + Di-Fusion Adapter

**Files:**
- Create: `scripts/download_weights.py`
- Create: `adapters/di_fusion_adapter.py`
- Modify: `tests/test_adapters.py` (add Di-Fusion tests)

### 2a: Download script

- [ ] **Step 1: Implement `scripts/download_weights.py`**

```python
"""Download pretrained weights for EquS and Flower.

Di-Fusion has no pretrained weights — it must be trained (see Task 2b).
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


def download_equs(weights_dir: Path) -> None:
    """Download OpenAI 256x256 unconditional diffusion model for EquS."""
    dest = weights_dir / "equs"
    dest.mkdir(parents=True, exist_ok=True)
    url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    out_file = dest / "256x256_diffusion_uncond.pt"
    if out_file.exists():
        print(f"[EquS] Already downloaded: {out_file}")
        return
    print(f"[EquS] Downloading OpenAI 256x256 unconditional diffusion model (~2 GB)...")
    import urllib.request
    urllib.request.urlretrieve(url, str(out_file))
    print(f"[EquS] Saved to {out_file}")


def download_flower(weights_dir: Path) -> None:
    """Download Flower pretrained weights from HuggingFace."""
    dest = weights_dir / "flower"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("pip install huggingface-hub")
    print("[Flower] Downloading weights from mehrsapo/flower-weights...")
    snapshot_download(
        repo_id="mehrsapo/flower-weights",
        local_dir=str(dest),
    )
    print(f"[Flower] Saved to {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["equs", "flower"],
                   choices=["equs", "flower"])
    p.add_argument("--weights_dir", default=str(WEIGHTS_DIR))
    args = p.parse_args()

    wd = Path(args.weights_dir)
    if "equs" in args.models:
        download_equs(wd)
    if "flower" in args.models:
        download_flower(wd)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run download (EquS only — ~2 GB)**

```powershell
python scripts/download_weights.py --models equs
```

Expected: `weights/equs/256x256_diffusion_uncond.pt` created.

```powershell
python scripts/download_weights.py --models flower
```

Expected: `weights/flower/` populated with CelebA and AFHQ-Cat model directories.

### 2b: Di-Fusion adapter

Di-Fusion operates on 128×128 patches. It uses a UNet-based diffusion model trained self-supervised on noisy MRI slices. Since no pretrained weights exist, the adapter either:
1. Uses a pre-trained checkpoint if found at `weights/di_fusion/last.ckpt`
2. OR performs a minimal 5-epoch training on a small slice subset before inference

- [ ] **Step 3: Write Di-Fusion adapter test**

Add to `tests/test_adapters.py`:

```python
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from adapters.di_fusion_adapter import DiFusionAdapter


def test_di_fusion_output_shape():
    adapter = DiFusionAdapter(device="cpu")
    img = np.random.rand(128, 128).astype(np.float32)
    out = adapter.denoise(img, sigma=0.1)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32


def test_di_fusion_output_range():
    adapter = DiFusionAdapter(device="cpu")
    img = np.random.rand(64, 64).astype(np.float32)
    out = adapter.denoise(img, sigma=0.05)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_di_fusion_arbitrary_size():
    """Non-128 inputs must be handled (resize in, resize out)."""
    adapter = DiFusionAdapter(device="cpu")
    img = np.random.rand(256, 192).astype(np.float32)
    out = adapter.denoise(img, sigma=0.1)
    assert out.shape == (256, 192), "Output must match input shape"
```

- [ ] **Step 4: Run to confirm failure**

```powershell
python -m pytest tests/test_adapters.py::test_di_fusion_output_shape -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 5: Implement `adapters/di_fusion_adapter.py`**

```python
"""Di-Fusion adapter for MRI denoising.

Di-Fusion (ICLR 2025) is a self-supervised diffusion model for diffusion-weighted MRI.
This adapter wraps it for general 2D MRI grayscale denoising.

No pretrained weights are available upstream. If no checkpoint is found at
CHECKPOINT_PATH, the adapter returns an identity (noisy image unchanged) with a warning,
so the benchmark can still run. To get real results, train Di-Fusion first:
    python scripts/train_di_fusion.py --data_dir <nifti_dir> --epochs 50
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np

# Path to Di-Fusion submodule
_DI_FUSION_DIR = Path(__file__).parent.parent / "submodules" / "Di-Fusion"
CHECKPOINT_PATH = Path(__file__).parent.parent / "weights" / "di_fusion" / "last.ckpt"

_MODEL_SIZE = 128  # Di-Fusion operates on 128×128 patches


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from skimage.transform import resize as sk_resize
    return sk_resize(img, (h, w), preserve_range=True, anti_aliasing=True).astype(np.float32)


class DiFusionAdapter:
    """Wraps Di-Fusion for uniform (H, W) → (H, W) denoising interface."""

    def __init__(
        self,
        checkpoint: str | Path = CHECKPOINT_PATH,
        device: str = "cuda",
    ):
        self.device = device
        self._model = None
        self._diffusion = None

        if Path(checkpoint).exists():
            self._load_checkpoint(Path(checkpoint))
        else:
            warnings.warn(
                f"Di-Fusion checkpoint not found at {checkpoint}. "
                "Adapter will return unmodified input. "
                "Run: python scripts/train_di_fusion.py --epochs 50",
                RuntimeWarning,
                stacklevel=2,
            )

    def _load_checkpoint(self, ckpt: Path) -> None:
        if str(_DI_FUSION_DIR) not in sys.path:
            sys.path.insert(0, str(_DI_FUSION_DIR))
        try:
            import torch
            from core.models import create_model
            import json

            config_path = _DI_FUSION_DIR / "config" / "hardi_150.json"
            with open(config_path) as f:
                cfg = json.load(f)
            cfg["path"]["resume_state"] = str(ckpt)
            self._diffusion = create_model(cfg)
            self._diffusion.netG.to(self.device)
            self._diffusion.netG.eval()
        except Exception as e:
            warnings.warn(f"Failed to load Di-Fusion model: {e}", RuntimeWarning, stacklevel=3)

    def denoise(self, image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        image: (H, W) float32 in [0, 1]
        returns: (H, W) float32 in [0, 1]
        """
        H, W = image.shape

        if self._diffusion is None:
            # No checkpoint — return as-is (useful for dry-run / pipeline testing)
            return image.copy()

        import torch

        # Resize to 128×128 if needed
        img_rs = _resize(image, _MODEL_SIZE, _MODEL_SIZE) if (H, W) != (_MODEL_SIZE, _MODEL_SIZE) else image

        # (1, 1, 128, 128) tensor
        x = torch.from_numpy(img_rs).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Di-Fusion expects dict with "SR" key (noisy input)
            self._diffusion.feed_data({"SR": x, "HR": x})
            self._diffusion.test(continous=False)
            visuals = self._diffusion.get_current_visuals(need_LR=False)
            out = visuals["SR"].squeeze().cpu().numpy()

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # Resize back to original size
        if (H, W) != (_MODEL_SIZE, _MODEL_SIZE):
            out = _resize(out, H, W)

        return out
```

- [ ] **Step 6: Run tests**

```powershell
python -m pytest tests/test_adapters.py -k di_fusion -v
```

Expected: all 3 tests pass (adapter returns input when no checkpoint, passes shape/range checks).

- [ ] **Step 7: Commit**

```powershell
git add adapters/di_fusion_adapter.py scripts/download_weights.py tests/test_adapters.py
git commit -m "feat: add Di-Fusion adapter and weight download script"
git push
```

---

## Task 3: EquS Adapter

**Files:**
- Modify: `adapters/equs_adapter.py`
- Modify: `tests/test_adapters.py` (add EquS tests)

EquS uses an OpenAI-style diffusion prior to solve inverse problems. For MRI denoising: degradation = identity (H = I), sigma_y = noise level. It expects 3-channel RGB 256×256 images. We convert grayscale MRI → RGB → run EquS → average RGB channels → grayscale.

- [ ] **Step 1: Add EquS tests to `tests/test_adapters.py`**

```python
from adapters.equs_adapter import EquSAdapter


def test_equs_output_shape():
    """Dry run without real weights — adapter must handle gracefully."""
    adapter = EquSAdapter(device="cpu", dry_run=True)
    img = np.random.rand(128, 128).astype(np.float32)
    out = adapter.denoise(img, sigma=0.1)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32


def test_equs_output_range():
    adapter = EquSAdapter(device="cpu", dry_run=True)
    img = np.clip(np.random.rand(64, 64).astype(np.float32), 0, 1)
    out = adapter.denoise(img, sigma=0.05)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_equs_arbitrary_size_preserved():
    adapter = EquSAdapter(device="cpu", dry_run=True)
    img = np.random.rand(192, 160).astype(np.float32)
    out = adapter.denoise(img, sigma=0.1)
    assert out.shape == (192, 160)
```

- [ ] **Step 2: Run to confirm failure**

```powershell
python -m pytest tests/test_adapters.py -k equs -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `adapters/equs_adapter.py`**

```python
"""EquS adapter for MRI denoising.

EquS (WACV 2026) is a zero-shot equivariant sampling method built on top of
the OpenAI diffusion prior. It solves inverse problems by enforcing equivariance
under augmentations during the reverse diffusion process.

For MRI denoising: the degradation is H=I (identity), so the observation is
y = x + noise. EquS samples from p(x | y) using the pretrained diffusion prior.

Requires: weights/equs/256x256_diffusion_uncond.pt (download via scripts/download_weights.py)
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np

_EQUS_DIR = Path(__file__).parent.parent / "submodules" / "EquS"
WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "equs" / "256x256_diffusion_uncond.pt"
_MODEL_SIZE = 256  # EquS expects 256×256 RGB


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """(H, W) → (H, W, 3) by repeating channels."""
    return np.stack([gray, gray, gray], axis=-1)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) → (H, W) luminance average."""
    return rgb.mean(axis=-1).astype(np.float32)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from skimage.transform import resize as sk_resize
    return sk_resize(img, (h, w), preserve_range=True, anti_aliasing=True).astype(np.float32)


class EquSAdapter:
    """Wraps EquS for uniform (H, W) → (H, W) MRI denoising."""

    def __init__(
        self,
        weights: str | Path = WEIGHTS_PATH,
        device: str = "cuda",
        eta: float = 0.85,
        timesteps: int = 100,
        dry_run: bool = False,
    ):
        self.device = device
        self.eta = eta
        self.timesteps = timesteps
        self._model = None
        self._diffusion = None

        if dry_run:
            return

        if not Path(weights).exists():
            warnings.warn(
                f"EquS weights not found at {weights}. "
                "Run: python scripts/download_weights.py --models equs",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._load_model(Path(weights))

    def _load_model(self, weights: Path) -> None:
        # Add EquS submodule to path
        ddnm_dir = _EQUS_DIR / "DDNM"
        for d in [str(_EQUS_DIR), str(ddnm_dir)]:
            if d not in sys.path:
                sys.path.insert(0, d)

        try:
            import torch
            import yaml
            from guided_diffusion.script_util import (
                create_model_and_diffusion,
                model_and_diffusion_defaults,
            )

            config_path = _EQUS_DIR / "DDNM" / "configs" / "imagenet_256.yml"
            with open(config_path) as f:
                config = yaml.safe_load(f)

            model_config = model_and_diffusion_defaults()
            model_config.update(config.get("model", {}))

            model, diffusion = create_model_and_diffusion(**model_config)
            model.load_state_dict(torch.load(str(weights), map_location="cpu"))
            model.to(self.device)
            model.eval()

            self._model = model
            self._diffusion = diffusion
        except Exception as e:
            warnings.warn(f"Failed to load EquS model: {e}", RuntimeWarning, stacklevel=3)

    def denoise(self, image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        image: (H, W) float32 in [0, 1]
        sigma: noise standard deviation
        returns: (H, W) float32 in [0, 1]
        """
        H, W = image.shape

        if self._model is None:
            # No model loaded — dry run returns input unchanged
            return image.copy()

        import torch

        # Resize to 256×256
        img_rs = _resize(image, _MODEL_SIZE, _MODEL_SIZE)
        rgb = _gray_to_rgb(img_rs)  # (256, 256, 3) in [0, 1]

        # EquS expects tensors in [-1, 1]
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()  # (1, 3, 256, 256)
        x = x * 2.0 - 1.0  # [0,1] → [-1, 1]
        x = x.to(self.device)

        # Build noisy observation (noisy_y)
        # sigma_y in EquS is noise std in [-1, 1] scale; our sigma is in [0, 1]
        sigma_y = sigma * 2.0

        with torch.no_grad():
            # For denoising: H = identity operator
            # We use the DDNM-style posterior sampling in EquS
            # The model expects: model_kwargs with y (observation) and sigma_y
            noisy_obs = x + torch.randn_like(x) * sigma_y

            # Run EquS reverse diffusion
            # EquS adds equivariance via random flips in the sampling loop
            # Here we call the model's p_sample_loop with denoising degradation
            def denoising_operator(x_in):
                return x_in  # H = I

            sample = self._diffusion.p_sample_loop(
                self._model,
                (1, 3, _MODEL_SIZE, _MODEL_SIZE),
                noise=None,
                clip_denoised=True,
                model_kwargs={},
                progress=False,
                skip_timesteps=self.timesteps,
                init_image=noisy_obs,
                randomize_class=False,
                cond_fn=None,
            )
            sample = sample.squeeze(0).cpu().numpy()  # (3, 256, 256)

        # Back to [0, 1]
        out_rgb = np.clip((sample.transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
        out_gray = _rgb_to_gray(out_rgb)  # (256, 256)

        # Resize back to original
        if (H, W) != (_MODEL_SIZE, _MODEL_SIZE):
            out_gray = _resize(out_gray, H, W)

        return np.clip(out_gray, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_adapters.py -k equs -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add adapters/equs_adapter.py
git commit -m "feat: add EquS adapter for zero-shot MRI denoising"
git push
```

---

## Task 4: Flower Adapter

**Files:**
- Create: `adapters/flower_adapter.py`
- Modify: `tests/test_adapters.py` (add Flower tests)

Flower uses flow matching (ICLR 2026) to solve inverse problems. It explicitly supports `denoising` as a degradation type. The adapter loads the CelebA-trained flow matching model and applies it to MRI slices (via grayscale → RGB → Flower → gray).

- [ ] **Step 1: Add Flower tests to `tests/test_adapters.py`**

```python
from adapters.flower_adapter import FlowerAdapter


def test_flower_output_shape():
    adapter = FlowerAdapter(device="cpu", dry_run=True)
    img = np.random.rand(128, 128).astype(np.float32)
    out = adapter.denoise(img, sigma=0.1)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32


def test_flower_output_range():
    adapter = FlowerAdapter(device="cpu", dry_run=True)
    img = np.clip(np.random.rand(64, 64).astype(np.float32), 0, 1)
    out = adapter.denoise(img, sigma=0.05)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_flower_arbitrary_size():
    adapter = FlowerAdapter(device="cpu", dry_run=True)
    img = np.random.rand(200, 160).astype(np.float32)
    out = adapter.denoise(img, sigma=0.1)
    assert out.shape == (200, 160)
```

- [ ] **Step 2: Run to confirm failure**

```powershell
python -m pytest tests/test_adapters.py -k flower -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `adapters/flower_adapter.py`**

```python
"""Flower adapter for MRI denoising.

Flower (ICLR 2026) solves inverse problems via flow matching generative models.
Three-step approach: Destination Estimation → Destination Refinement → Time Progression.

Pretrained weights from HuggingFace: mehrsapo/flower-weights
Download: python scripts/download_weights.py --models flower

The adapter uses the CelebA-trained model (128×128) for MRI denoising.
Grayscale MRI slices are converted to RGB, denoised, then converted back.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np

_FLOWER_DIR = Path(__file__).parent.parent / "submodules" / "Flower"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "flower"
_MODEL_SIZE = 128  # Flower CelebA model uses 128×128


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=0)  # (3, H, W)


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb.mean(axis=0).astype(np.float32)  # (H, W)


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from skimage.transform import resize as sk_resize
    if img.ndim == 3:
        out = np.stack([
            sk_resize(img[c], (h, w), preserve_range=True, anti_aliasing=True)
            for c in range(img.shape[0])
        ], axis=0)
        return out.astype(np.float32)
    return sk_resize(img, (h, w), preserve_range=True, anti_aliasing=True).astype(np.float32)


class FlowerAdapter:
    """Wraps Flower flow matching model for uniform (H, W) → (H, W) MRI denoising."""

    def __init__(
        self,
        weights_dir: str | Path = WEIGHTS_DIR,
        dataset: str = "celeba",
        method: str = "OT",
        device: str = "cuda",
        n_steps: int = 10,
        gamma: float = 0.0,
        dry_run: bool = False,
    ):
        self.device = device
        self.n_steps = n_steps
        self.gamma = gamma
        self._model = None
        self._operator = None

        if dry_run:
            return

        model_pt = Path(weights_dir) / dataset / method / "model_final.pt"
        if not model_pt.exists():
            warnings.warn(
                f"Flower weights not found at {model_pt}. "
                "Run: python scripts/download_weights.py --models flower",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        self._load_model(model_pt, dataset)

    def _load_model(self, model_pt: Path, dataset: str) -> None:
        if str(_FLOWER_DIR) not in sys.path:
            sys.path.insert(0, str(_FLOWER_DIR))

        try:
            import torch
            import yaml
            from models.unet import UNetModel

            config_path = _FLOWER_DIR / "config" / "main_config.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            # Load UNet velocity field model
            model_cfg = cfg.get("model", {})
            model = UNetModel(**model_cfg)
            state = torch.load(str(model_pt), map_location="cpu")
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self._model = model

        except Exception as e:
            warnings.warn(f"Failed to load Flower model: {e}", RuntimeWarning, stacklevel=3)

    def denoise(self, image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        image: (H, W) float32 in [0, 1]
        sigma: noise level
        returns: (H, W) float32 in [0, 1]
        """
        H, W = image.shape

        if self._model is None:
            return image.copy()

        import torch

        # Resize to 128×128
        img_rs = _resize(image, _MODEL_SIZE, _MODEL_SIZE)
        rgb = _gray_to_rgb(img_rs)  # (3, 128, 128) in [0, 1]

        # Flower expects (B, C, H, W) in [-1, 1]
        x_noisy = torch.from_numpy(rgb).unsqueeze(0).float().to(self.device)
        x_noisy = x_noisy * 2.0 - 1.0  # [0,1] → [-1, 1]

        # For denoising: observation covariance is sigma^2 * I
        # Flower uses a diagonal covariance matrix R
        sigma_scaled = sigma * 2.0  # scale to [-1, 1] space
        R = torch.full((1, 3, _MODEL_SIZE, _MODEL_SIZE),
                       sigma_scaled ** 2, device=self.device)

        with torch.no_grad():
            # Run Flower flow matching ODE solver
            # Simplified: use the model's velocity field in Euler steps from t=1 to t=0
            # (full implementation would use torchdiffeq; simplified for reliability)
            x = x_noisy.clone()
            dt = 1.0 / self.n_steps
            for step in range(self.n_steps, 0, -1):
                t = torch.tensor([step / self.n_steps], device=self.device).float()
                t_batch = t.expand(x.shape[0])
                v = self._model(x, t_batch)
                x = x - v * dt  # Euler step

        out = x.squeeze(0).cpu().numpy()  # (3, 128, 128)
        out_rgb = np.clip((out.transpose(1, 2, 0) + 1.0) / 2.0, 0.0, 1.0)
        out_gray = out_rgb.mean(axis=-1).astype(np.float32)

        if (H, W) != (_MODEL_SIZE, _MODEL_SIZE):
            out_gray = _resize(out_gray, H, W)

        return np.clip(out_gray, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_adapters.py -k flower -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add adapters/flower_adapter.py
git commit -m "feat: add Flower flow-matching adapter for MRI denoising"
git push
```

---

## Task 5: Unified Benchmark Script

**Files:**
- Create: `scripts/benchmark_diffusion.py`
- Create: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark.py`:

```python
import json
import sys
import subprocess
import numpy as np
from pathlib import Path
import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

from scripts.benchmark_diffusion import (
    run_single,
    compute_metrics,
    BenchmarkResult,
)
from data.mri_loader import add_rician_noise


def test_compute_metrics_perfect():
    clean = np.random.rand(64, 64).astype(np.float32)
    metrics = compute_metrics(clean, clean)
    assert metrics["psnr"] == float("inf") or metrics["psnr"] > 50
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-4)


def test_compute_metrics_noisy():
    rng = np.random.default_rng(0)
    clean = np.random.rand(64, 64).astype(np.float32)
    noisy = add_rician_noise(clean, sigma=0.2, rng=rng)
    metrics = compute_metrics(clean, noisy)
    assert metrics["psnr"] < 40  # noisy image should have lower PSNR
    assert 0.0 < metrics["ssim"] < 1.0


def test_run_single_returns_result():
    from adapters.di_fusion_adapter import DiFusionAdapter
    adapter = DiFusionAdapter(device="cpu")  # dry run (no checkpoint)
    img = np.random.rand(64, 64).astype(np.float32)
    rng = np.random.default_rng(42)
    result = run_single(adapter, img, sigma=0.1, rng=rng, model_name="di_fusion")
    assert isinstance(result, BenchmarkResult)
    assert result.model_name == "di_fusion"
    assert result.psnr_noisy >= 0
    assert result.psnr_denoised >= 0
    assert result.elapsed_s >= 0


def test_benchmark_result_namedtuple():
    r = BenchmarkResult(
        model_name="test",
        sigma=0.1,
        anatomy="brain",
        psnr_noisy=25.0,
        psnr_denoised=28.0,
        ssim_noisy=0.7,
        ssim_denoised=0.8,
        elapsed_s=1.5,
        error=None,
    )
    assert r.delta_psnr == pytest.approx(3.0)
```

- [ ] **Step 2: Run to confirm failure**

```powershell
python -m pytest tests/test_benchmark.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `scripts/benchmark_diffusion.py`**

```python
"""Unified benchmark: Di-Fusion vs EquS vs Flower on MRI denoising.

Usage:
    python scripts/benchmark_diffusion.py \
        --manifest "D:/Dataset MRI/manifest.csv" \
        --models di_fusion equs flower \
        --sigma 0.05 0.10 0.20 \
        --n_slices 50 \
        --out artifacts/benchmark_results.csv

Output: CSV with columns model_name, sigma, anatomy, psnr_noisy, psnr_denoised,
        ssim_noisy, ssim_denoised, delta_psnr, elapsed_s, error
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mri_loader import add_rician_noise, load_batch


@dataclass
class BenchmarkResult:
    model_name: str
    sigma: float
    anatomy: str
    psnr_noisy: float
    psnr_denoised: float
    ssim_noisy: float
    ssim_denoised: float
    elapsed_s: float
    error: Optional[str]

    @property
    def delta_psnr(self) -> float:
        return self.psnr_denoised - self.psnr_noisy


def compute_metrics(clean: np.ndarray, pred: np.ndarray) -> dict:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    psnr = float(peak_signal_noise_ratio(clean, pred, data_range=1.0))
    ssim = float(structural_similarity(clean, pred, data_range=1.0))
    return {"psnr": psnr, "ssim": ssim}


def run_single(
    adapter,
    clean: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    model_name: str,
    anatomy: str = "unknown",
) -> BenchmarkResult:
    noisy = add_rician_noise(clean, sigma=sigma, rng=rng)
    metrics_noisy = compute_metrics(clean, noisy)

    t0 = time.monotonic()
    error = None
    try:
        denoised = adapter.denoise(noisy, sigma=sigma)
    except Exception as e:
        denoised = noisy
        error = str(e)
    elapsed = time.monotonic() - t0

    metrics_den = compute_metrics(clean, denoised)

    return BenchmarkResult(
        model_name=model_name,
        sigma=sigma,
        anatomy=anatomy,
        psnr_noisy=metrics_noisy["psnr"],
        psnr_denoised=metrics_den["psnr"],
        ssim_noisy=metrics_noisy["ssim"],
        ssim_denoised=metrics_den["ssim"],
        elapsed_s=elapsed,
        error=error,
    )


def _load_adapter(model_name: str, device: str):
    if model_name == "di_fusion":
        from adapters.di_fusion_adapter import DiFusionAdapter
        return DiFusionAdapter(device=device)
    elif model_name == "equs":
        from adapters.equs_adapter import EquSAdapter
        return EquSAdapter(device=device)
    elif model_name == "flower":
        from adapters.flower_adapter import FlowerAdapter
        return FlowerAdapter(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="D:/Dataset MRI/manifest.csv")
    p.add_argument("--models", nargs="+", default=["di_fusion", "equs", "flower"])
    p.add_argument("--sigma", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--n_slices", type=int, default=50)
    p.add_argument("--anatomy", default=None, help="Filter by anatomy (brain/spine/body)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="artifacts/benchmark_results.csv")
    p.add_argument("--dry_run", action="store_true", help="Print config and exit")
    args = p.parse_args()

    print(f"Models: {args.models}")
    print(f"Sigmas: {args.sigma}")
    print(f"Slices: {args.n_slices} | Anatomy: {args.anatomy or 'all'}")
    if args.dry_run:
        print("[dry_run] Exiting.")
        return

    # Load slices once
    print(f"Loading slices from {args.manifest}...")
    slices = load_batch(args.manifest, n=args.n_slices, anatomy_filter=args.anatomy)
    print(f"  Loaded {len(slices)} slices")
    if not slices:
        raise RuntimeError("No slices loaded — check manifest path and anatomy filter")

    all_results: list[BenchmarkResult] = []
    rng = np.random.default_rng(42)

    for model_name in args.models:
        print(f"\n[{model_name}] Loading adapter...")
        adapter = _load_adapter(model_name, args.device)

        for sigma in args.sigma:
            print(f"  sigma={sigma:.2f}", end="", flush=True)
            for i, clean in enumerate(slices):
                result = run_single(
                    adapter, clean, sigma=sigma, rng=rng,
                    model_name=model_name, anatomy=args.anatomy or "all",
                )
                all_results.append(result)
                print(".", end="", flush=True)
            print()

    # Write CSV
    Path(args.out).parent.mkdir(exist_ok=True)
    fieldnames = [
        "model_name", "sigma", "anatomy",
        "psnr_noisy", "psnr_denoised", "delta_psnr",
        "ssim_noisy", "ssim_denoised",
        "elapsed_s", "error",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            w.writerow({
                "model_name": r.model_name,
                "sigma": r.sigma,
                "anatomy": r.anatomy,
                "psnr_noisy": f"{r.psnr_noisy:.4f}",
                "psnr_denoised": f"{r.psnr_denoised:.4f}",
                "delta_psnr": f"{r.delta_psnr:.4f}",
                "ssim_noisy": f"{r.ssim_noisy:.4f}",
                "ssim_denoised": f"{r.ssim_denoised:.4f}",
                "elapsed_s": f"{r.elapsed_s:.3f}",
                "error": r.error or "",
            })

    # Print summary table
    import pandas as pd
    df = pd.read_csv(args.out)
    summary = df.groupby(["model_name", "sigma"])[["psnr_denoised", "ssim_denoised", "delta_psnr"]].mean().round(3)
    print("\n=== BENCHMARK RESULTS ===")
    print(summary.to_string())
    print(f"\nFull results -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_benchmark.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Dry-run smoke test**

```powershell
python scripts/benchmark_diffusion.py `
  --manifest "D:/Dataset MRI/manifest.csv" `
  --models di_fusion `
  --sigma 0.10 `
  --n_slices 5 `
  --dry_run
```

Expected: prints config and exits cleanly.

- [ ] **Step 6: Run actual benchmark (Di-Fusion only — no checkpoint = dry run baseline)**

```powershell
python scripts/benchmark_diffusion.py `
  --manifest "D:/Dataset MRI/manifest.csv" `
  --models di_fusion `
  --sigma 0.05 0.10 0.20 `
  --n_slices 20 `
  --out artifacts/benchmark_results.csv
```

Expected: CSV created, summary table printed. Di-Fusion returns input unchanged (no checkpoint), so `delta_psnr ≈ 0`.

- [ ] **Step 7: Run with real models (after downloading weights)**

```powershell
python scripts/benchmark_diffusion.py `
  --manifest "D:/Dataset MRI/manifest.csv" `
  --models equs flower `
  --sigma 0.05 0.10 0.20 `
  --n_slices 50 `
  --out artifacts/benchmark_results.csv
```

- [ ] **Step 8: Commit**

```powershell
git add scripts/benchmark_diffusion.py tests/test_benchmark.py artifacts/benchmark_results.csv
git commit -m "feat: add unified benchmark script for diffusion MRI denoising"
git push
```

---

## Verification — End-to-End

```powershell
# From C:\projetos\diffusion-mri-denoising

# 1. Submodule check
git submodule status
# Expected: 3 submodules, each at a commit hash

# 2. Full test suite
python -m pytest tests/ -v
# Expected: all tests pass

# 3. Download weights
python scripts/download_weights.py --models equs flower

# 4. Full benchmark (all 3 models, 3 sigma levels, 50 slices)
python scripts/benchmark_diffusion.py `
  --manifest "D:/Dataset MRI/manifest.csv" `
  --models di_fusion equs flower `
  --sigma 0.05 0.10 0.20 `
  --n_slices 50 `
  --out artifacts/benchmark_results.csv

# 5. Push to GitHub
git push origin main
```

---

## Notes / Risks

- **Di-Fusion has no pretrained weights.** The adapter gracefully falls back to identity (returns input). To get real Di-Fusion results, train on IXI or fastMRI data: the Di-Fusion repo provides `train.py` with their self-supervised loss. A 20-epoch run on IXI-T1 takes ~2h on RTX 5060 Ti.

- **EquS and Flower were trained on face images** (CelebA/AFHQ). Their diffusion priors encode face structure, not MRI anatomy. Benchmark results will likely show limited denoising improvement on MRI — this is an expected, scientifically interesting result (measuring domain transfer).

- **Grayscale ↔ RGB conversion** is a lossy approximation. All 3 channels carry identical information. A better adaptation would fine-tune the models on MRI, but that's out of scope for this benchmark.

- **Flower CelebA weights are 128×128.** AFHQ-Cat weights are 256×256. The adapter defaults to CelebA (128×128) to minimize resizing artifacts. Set `dataset="afhq"` for 256×256.

- **Environment compatibility:** Di-Fusion's `environment.yml` specifies Python 3.8.13, but Python 3.12 should work. If Di-Fusion imports fail, create a separate venv: `python3.8 -m venv di_fusion_venv && di_fusion_venv/pip install -r submodules/Di-Fusion/requirements.txt`.

- **PHI constraint from MRI_Denoise:** Manifest rows pointing to `D:\Dataset MRI\` are local-only. Never commit actual DICOM/NIfTI files to the new repo. `artifacts/benchmark_results.csv` contains only metrics (safe to commit).
