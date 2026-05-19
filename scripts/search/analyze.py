# MRI_Denoise/scripts/search/analyze.py
"""Read results.jsonl and print ranked best configs.

Usage:
    python -m scripts.search.analyze
    python -m scripts.search.analyze --file path/to/results.jsonl --top 10
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_FILE = Path("C:/projetos/benchmark_results/search/results.jsonl")


def load_results(path: Path) -> list[dict]:
    """Load all non-error rows from a JSONL results file."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            psnr = obj.get("psnr")
            # Skip error rows (psnr=None) and NaN rows
            if psnr is None:
                continue
            if isinstance(psnr, float) and math.isnan(psnr):
                continue
            rows.append(obj)
        except (json.JSONDecodeError, ValueError):
            pass
    return rows


def rank(rows: list[dict], top: int = 10) -> None:
    """Print a ranked table per (sigma, mode) and a cross-sigma best summary."""
    # Group by (sigma, mode)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["sigma"], r["mode"])].append(r)

    for (sigma, mode), group in sorted(groups.items()):
        group.sort(key=lambda x: x["psnr"], reverse=True)
        print(f"\n{'='*72}")
        print(f"  sigma={sigma}  mode={mode}  ({len(group)} trials)")
        print(f"{'='*72}")
        print(f"  {'#':>3}  {'PSNR':>7}  {'SSIM':>6}  {'model':22}  "
              f"{'gmap':13}  {'unsharp':12}  dither")
        print(f"  {'-'*3}  {'-'*7}  {'-'*6}  {'-'*22}  {'-'*13}  {'-'*12}  {'-'*20}")
        for i, r in enumerate(group[:top], 1):
            usharp = r["unsharp_cfg"]["name"]
            if usharp == "gsum":
                usharp += f"@{r['unsharp_cfg']['intensity']}"
            elif usharp == "unsharp":
                usharp += f"@{r['unsharp_cfg']['amount']}"
            elif usharp == "mlvum":
                usharp += f"@{r['unsharp_cfg']['scale']}"
            dstr = r["dither_cfg"]["name"]
            if dstr != "none":
                dstr += f"@{r['dither_cfg']['strength']}"
            ssim_str = f"{r['ssim']:.4f}" if r.get("ssim") is not None else "  N/A"
            print(f"  {i:>3}  {r['psnr']:7.3f}  {ssim_str}  "
                  f"{r['model_name']:22}  {r['gmap_strategy']:13}  "
                  f"{usharp:12}  {dstr}")

    # Cross-sigma summary: best model per sigma
    print(f"\n{'='*72}")
    print("  BEST PER SIGMA (any mode, any config)")
    print(f"{'='*72}")
    by_sigma: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        by_sigma[r["sigma"]].append(r)
    for sigma, group in sorted(by_sigma.items()):
        best = max(group, key=lambda x: x["psnr"])
        print(f"  sigma={sigma}: PSNR={best['psnr']:.3f} dB  "
              f"model={best['model_name']}  mode={best['mode']}  "
              f"gmap={best['gmap_strategy']}  "
              f"unsharp={best['unsharp_cfg']['name']}  "
              f"dither={best['dither_cfg']['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank denoiser search results")
    parser.add_argument("--file", type=str, default=str(DEFAULT_FILE))
    parser.add_argument("--top",  type=int, default=10)
    args = parser.parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"Results file not found: {path}")
        raise SystemExit(1)
    rows = load_results(path)
    print(f"Loaded {len(rows)} valid results from {path}")
    rank(rows, top=args.top)
