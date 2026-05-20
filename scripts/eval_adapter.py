"""Evaluate and compare baseline vs LoRA vs FouRA on held-out test split.

Usage:
    python scripts/eval_adapter.py \\
        --split data/finetune_split.json \\
        --baseline_weights weights/nafnet_pretrained.pth \\
        --lora_weights experiments/lora_nafnet_mri_finetune/checkpoints/best.pth \\
        --foura_weights experiments/foura_nafnet_mri_finetune/checkpoints/best.pth \\
        --out artifacts/adapter_benchmark.csv
"""
from __future__ import annotations
import argparse, json, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def load_dicom_slice(file_path: str) -> np.ndarray:
    """Load a DICOM file and return a float32 array normalized to [0, 1]."""
    import pydicom
    ds = pydicom.dcmread(file_path)
    pixel = ds.pixel_array.astype(np.float32)
    pixel_min, pixel_max = pixel.min(), pixel.max()
    if pixel_max > pixel_min:
        pixel = (pixel - pixel_min) / (pixel_max - pixel_min)
    return pixel


def add_rician_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Rician noise: magnitude = sqrt((img + n1)^2 + n2^2)."""
    n1 = np.random.randn(*img.shape).astype(np.float32) * sigma
    n2 = np.random.randn(*img.shape).astype(np.float32) * sigma
    return np.sqrt((img + n1) ** 2 + n2 ** 2).astype(np.float32)


def evaluate_model(model: torch.nn.Module, test_records: list, device: torch.device,
                   sigma: float = 0.10) -> pd.DataFrame:
    model.eval()
    results = []
    with torch.no_grad():
        for rec in test_records:
            fp = rec.get("file_path", "")
            if not Path(fp).exists():
                logger.warning("Skipping missing file: %s", fp)
                continue
            try:
                img = load_dicom_slice(fp)
                noisy = add_rician_noise(img, sigma)
                sigma_map = np.full_like(img, sigma, dtype=np.float32)
                inp = torch.from_numpy(
                    np.stack([noisy, sigma_map], axis=0)[None]
                ).to(device)
                pred = model(inp).squeeze().cpu().numpy()
                pred = np.clip(pred, 0.0, 1.0)
                results.append({
                    "psnr": psnr_metric(img, pred, data_range=1.0),
                    "ssim": ssim_metric(img, pred, data_range=1.0),
                    "anatomy": rec.get("anatomy", "unknown"),
                    "vendor": rec.get("vendor", "unknown"),
                })
            except Exception as e:
                logger.warning("Error processing %s: %s", fp, e)
    return pd.DataFrame(results)


def build_model(base_cfg: dict, weights_path: str, adapter_type: str | None,
                adapter_cfg: dict, device: torch.device) -> torch.nn.Module:
    from src.models.nafnet import NAFNet
    model = NAFNet(
        img_channel=base_cfg.get("in_channels", 2),
        width=base_cfg.get("width", 64),
        enc_blk_nums=base_cfg.get("enc_blk_nums", [2, 2, 4, 8]),
        middle_blk_num=base_cfg.get("middle_blk_num", 12),
        dec_blk_nums=base_cfg.get("dec_blk_nums", [2, 2, 2, 2]),
    )
    if adapter_type == "lora":
        for p in model.parameters():
            p.requires_grad = False
        from src.models.lora_adapter import attach_lora
        model = attach_lora(model, rank=adapter_cfg.get("rank", 16),
                            alpha=float(adapter_cfg.get("alpha", 32.0)))
    elif adapter_type == "foura":
        from src.models.foura_adapter import FouRAWrapper
        model = FouRAWrapper(model, rank=adapter_cfg.get("rank", 32),
                             alpha=float(adapter_cfg.get("alpha", 64.0)))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="data/finetune_split.json")
    p.add_argument("--baseline_weights", default=None)
    p.add_argument("--lora_weights", default=None)
    p.add_argument("--foura_weights", default=None)
    p.add_argument("--out", default="artifacts/adapter_benchmark.csv")
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit test samples for quick evaluation")
    args = p.parse_args()

    split_path = Path(args.split)
    if not split_path.exists():
        print(f"ERROR: Split file not found: {args.split}")
        print("Run: python scripts/prepare_finetune_dataset.py --out data/finetune_split.json")
        sys.exit(1)

    split = json.loads(split_path.read_text())
    test_records = split["test"]
    if args.max_samples:
        test_records = test_records[:args.max_samples]
    print(f"Evaluating on {len(test_records)} test samples (sigma={args.sigma})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = dict(in_channels=2, width=64,
                    enc_blk_nums=[2,2,4,8], middle_blk_num=12, dec_blk_nums=[2,2,2,2])

    variants = [
        ("Baseline", args.baseline_weights, None, {}),
        ("LoRA",     args.lora_weights,     "lora",  {"rank": 16, "alpha": 32.0}),
        ("FouRA",    args.foura_weights,    "foura", {"rank": 32, "alpha": 64.0}),
    ]

    rows = []
    for variant_name, weights_path, adapter_type, adapter_cfg in variants:
        if not weights_path:
            print(f"[SKIP] {variant_name}: no weights path provided")
            continue
        if not Path(weights_path).exists():
            print(f"[SKIP] {variant_name}: weights not found at {weights_path}")
            continue
        print(f"\nEvaluating {variant_name}...")
        model = build_model(base_cfg, weights_path, adapter_type, adapter_cfg, device)
        df = evaluate_model(model, test_records, device, sigma=args.sigma)
        if df.empty:
            print(f"  No results for {variant_name}")
            continue
        df["variant"] = variant_name
        rows.append(df)
        print(f"  PSNR: {df['psnr'].mean():.2f} dB  SSIM: {df['ssim'].mean():.4f}")

    if not rows:
        print("\nNo results to report. Provide at least one --*_weights argument.")
        sys.exit(0)

    results = pd.concat(rows, ignore_index=True)
    summary = results.groupby(["variant", "anatomy"])[["psnr", "ssim"]].mean().round(3)
    print("\n=== Summary ===")
    print(summary.to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"\nFull results saved → {args.out}")


if __name__ == "__main__":
    main()
