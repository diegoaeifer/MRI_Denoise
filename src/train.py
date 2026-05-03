"""
Thin shim — the MONAI-first training entry point is in mri_denoise.train.

Usage:
    python -m mri_denoise.train --config src/mri_denoise/configs/train.yaml
    python -m mri_denoise.train --config src/mri_denoise/configs/finetune_swinunetr.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from mri_denoise.train import main, load_config  # noqa: F401

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MRI Denoising — MONAI training")
    parser.add_argument("--config", default="src/mri_denoise/configs/train.yaml")
    main(parser.parse_args())
