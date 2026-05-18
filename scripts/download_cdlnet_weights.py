"""Download CDLNet blind-denoising pretrained weights.

Paper: CDLNet (IEEE OJSP 2022) https://ieeexplore.ieee.org/document/9769957
Repo:  https://github.com/nikopj/CDLNET-OJSP

Usage:
    python scripts/download_cdlnet_weights.py
    python scripts/download_cdlnet_weights.py --url <custom-url>
"""
import argparse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_DEST = _ROOT / "weights" / "CDLNet"
_FILENAME = "CDLNet_blind_gray_K20_M64_P7.pth"
# Check https://github.com/nikopj/CDLNET-OJSP/releases for current URL
_DEFAULT_URL = (
    "https://github.com/nikopj/CDLNET-OJSP/releases/download/v1.0/"
    + _FILENAME
)


def _progress(count, block_size, total):
    if total > 0:
        pct = min(count * block_size / total * 100, 100)
        print(f"\r  {pct:.0f}%", end="", flush=True)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, str(dest), _progress)
        print(f"\nSaved ({dest.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Manual steps:")
        print("  1. Visit https://github.com/nikopj/CDLNET-OJSP")
        print("  2. Find pretrained_models/ or Releases")
        print(f"  3. Save the blind-denoising .pth to {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=_DEFAULT_URL)
    ap.add_argument("--dest", default=str(_DEST / _FILENAME))
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    download(args.url, dest)
    print(f'\nFactory config: "cdlnet": {{"weights_path": "{dest}", '
          '"K": 20, "M": 64, "P": 7, "s": 1, "adaptive": true, "init": false}')


if __name__ == "__main__":
    main()
