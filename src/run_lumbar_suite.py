"""
==========================================================
Lumbar Spine Experiment Suite
==========================================================
Runs a sequence of experiments with DeepInverse pretrained
models (2-channel adapted via ChannelAdapter) on 1000 DICOM
images from the RSNA Lumbar Spine dataset.

Each experiment auto-detects available VRAM and maximises
batch size via binary search before training begins.

Usage (after CUDA is installed):
    python src/run_lumbar_suite.py

Results land in:
    experiments/logs/lumbar_suite/<run_id>/
    experiments/checkpoints/lumbar_suite/<run_id>/
==========================================================
"""

import os
import sys
import subprocess
import datetime
import json
import logging
import time

# ------------------------------------------------------------------ #
#  Paths
# ------------------------------------------------------------------ #
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = r"C:\projetos\Lumbar_spine\rsna-2024-lumbar-spine-degenerative-classification\train_images"
BASE_CONFIG = "configs/config_lumbar_suite.yaml"
OUTPUT_DIR = "experiments"
LIMIT = 1000  # First 1000 images

# ------------------------------------------------------------------ #
#  Experiment matrix — pretrained deepinv models first, then scratch
# ------------------------------------------------------------------ #
EXPERIMENTS = [
    {
        "name": "restormer_pretrained",
        "model": "restormer",
        "epochs": 25,
        "batch": 4,
        "lr": "5e-5",
        "note": "DeepInv Restormer pretrained (3-ch adaptation)",
    },
    {
        "name": "swinir_pretrained",
        "model": "swinir_pretrained",
        "epochs": 25,
        "batch": 4,
        "lr": "5e-5",
        "note": "DeepInv SwinIR pretrained (3-ch adaptation)",
    },
]

# ------------------------------------------------------------------ #
#  Logging setup  (must create log dir first!)
# ------------------------------------------------------------------ #
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(OUTPUT_DIR, "lumbar_suite_orchestrator.log"), mode="w"
        ),
    ],
)
log = logging.getLogger("suite")


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def get_gpu_vram_gb() -> float:
    """Returns total VRAM in GB, or 0 if no CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass
    return 0.0


def suggest_batch_size(model_name: str, vram_gb: float, default: int) -> int:
    """
    Very rough heuristic: scale batch relative to a 12 GB baseline.
    Heavy models (restormer, scunet) get smaller slices of VRAM.
    """
    heavy_models = {"scunet_pretrained", "restormer", "gsdrunet", "swinir_pretrained"}
    light_models = {"dncnn_pretrained"}

    scale = vram_gb / 12.0  # normalised to 12 GB
    if model_name in heavy_models:
        scale *= 0.5
    elif model_name in light_models:
        scale *= 1.5

    suggested = max(1, int(default * scale))
    # Round to nearest power of 2 for cache efficiency
    p = 1
    while p * 2 <= suggested:
        p *= 2
    return p


def write_experiment_log(exp: dict, status: str, elapsed: float, log_path: str):
    record = {
        "experiment": exp["name"],
        "model": exp["model"],
        "status": status,
        "elapsed_min": round(elapsed / 60, 2),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    # PROJECT_ROOT is the cwd for all subprocess calls (so src/train.py resolves correctly)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_log = os.path.join(OUTPUT_DIR, "lumbar_suite_summary.jsonl")

    vram = get_gpu_vram_gb()
    log.info("=== Lumbar Spine Experiment Suite ===")
    log.info(f"GPU VRAM detected: {vram:.1f} GB")
    log.info(f"Total experiments: {len(EXPERIMENTS)}")
    log.info(f"Limit per split  : {LIMIT} images")
    log.info(f"Data path        : {DATA_DIR}")
    log.info("")

    suite_results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        # ---- Determine batch size --------------------------------- #
        bs = (
            suggest_batch_size(exp["model"], vram, exp["batch"])
            if vram > 0
            else exp["batch"]
        )
        log.info(f"[{i}/{len(EXPERIMENTS)}] Starting: {exp['name']}")
        log.info(
            f"  Model   : {exp['model']}  |  Epochs: {exp['epochs']}  |  Batch: {bs}  |  LR: {exp['lr']}"
        )
        log.info(f"  Note    : {exp['note']}")

        # ---- Build command ---------------------------------------- #
        # train.py accepts only ONE --config; we write a merged override yaml
        override_cfg_path = os.path.join(
            PROJECT_ROOT, f"configs/_exp_{exp['name']}_override.yaml"
        )
        with open(override_cfg_path, "w") as f:
            f.write(
                f"training:\n"
                f"  epochs: {exp['epochs']}\n"
                f"  batch_size: {bs}\n"
                f"  learning_rate: {exp['lr']}\n"
            )

        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "src", "mri_denoise", "train.py"),
            "--config",
            override_cfg_path,
        ]

        log_path = os.path.join(
            PROJECT_ROOT, OUTPUT_DIR, "lumbar_suite", f"{exp['name']}_stdout.log"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # ---- Run experiment --------------------------------------- #
        t0 = time.time()
        status = "SUCCESS"
        try:
            with open(log_path, "w") as logf:
                subprocess.run(
                    cmd,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=PROJECT_ROOT,  # always run from project root
                )
        except subprocess.CalledProcessError as e:
            status = f"FAILED (exit {e.returncode})"
            log.error(f"  FAILED: {exp['name']} --- see {log_path}")
        except Exception as e:
            status = f"ERROR: {e}"
            log.error(f"  ERROR in {exp['name']}: {e}")

        elapsed = time.time() - t0
        log.info(f"  -> {status} in {elapsed/60:.1f} min")
        log.info("")

        write_experiment_log(exp, status, elapsed, summary_log)
        suite_results.append(
            {"exp": exp["name"], "status": status, "min": round(elapsed / 60, 1)}
        )

        # Clean up temp override
        if os.path.exists(override_cfg_path):
            os.remove(override_cfg_path)

    # ---- Final summary ------------------------------------------- #
    log.info("=" * 60)
    log.info("SUITE COMPLETE – Summary")
    log.info("=" * 60)
    for r in suite_results:
        log.info(f"  {r['exp']:35s}  {r['status']:15s}  {r['min']} min")
    log.info(
        f"\nFull logs: {os.path.abspath(os.path.join(OUTPUT_DIR, 'lumbar_suite'))}"
    )
    log.info(f"JSONL summary: {os.path.abspath(summary_log)}")


if __name__ == "__main__":
    main()
