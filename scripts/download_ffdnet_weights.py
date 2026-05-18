"""Download FFDNet pretrained grayscale weights from KAIR model zoo.

Repo: https://github.com/cszn/KAIR
Weights: ffdnet_gray.pth (grayscale, in_nc=1, nc=64, nb=15)

Usage:
    python scripts/download_ffdnet_weights.py
    python scripts/download_ffdnet_weights.py --dest weights/FFDNet/ffdnet_gray.pth
"""
import argparse
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_DEST = _ROOT / "weights" / "FFDNet" / "ffdnet_gray.pth"
# KAIR model zoo — check https://github.com/cszn/KAIR/releases for current URL
_DEFAULT_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth"
)


def _progress(count, block_size, total):
    if total > 0:
        pct = min(count * block_size / total * 100, 100)
        print(f"\r  {pct:.0f}%", end="", flush=True)


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, str(dest), _progress)
        print(f"\nSaved ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Manual fallback:")
        print("  1. Visit https://github.com/cszn/KAIR/releases")
        print("  2. Download ffdnet_gray.pth")
        print(f"  3. Save to {dest}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Download FFDNet-KAIR weights")
    ap.add_argument("--url", default=_DEFAULT_URL)
    ap.add_argument("--dest", default=str(_DEST))
    args = ap.parse_args()
    dest = Path(args.dest)
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    ok = download(args.url, dest)
    if ok:
        print(f'\nFactory config: "ffdnet_kair": {{"weights_path": "{dest}"}}')


if __name__ == "__main__":
    main()
