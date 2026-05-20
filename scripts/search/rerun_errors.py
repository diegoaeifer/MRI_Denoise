"""Re-run specific error-prone task categories from the benchmark search.

Targets the 3 error categories found in the original results.jsonl:
  1. gsdrunet 2D (fixed: out_channels=1)
  2. snraware_* 3D (fixed: apply_model now handles 3D via patching)
  3. imt-mrd_* 3D (fixed: wrapper rewritten with correct input contracts)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_MRI_ROOT = Path(__file__).resolve().parents[2]

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from search.task_grid import (
    SIGMAS, GMAP_STRATEGIES, UNSHARP_CFGS, DITHER_CFGS
)
from search.eval_pipeline import run_trial

OUT_PATH = Path("C:/projetos/benchmark_results/search/results.jsonl")


def _task_hash(task: dict) -> str:
    key = json.dumps(task, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def build_error_tasks() -> list[dict]:
    tasks = []
    import itertools

    # Error category 1: gsdrunet in 2D mode (state_dict mismatch, now fixed)
    for sigma, gmap, unsharp, dither in itertools.product(
        SIGMAS, GMAP_STRATEGIES, UNSHARP_CFGS, DITHER_CFGS
    ):
        tasks.append({
            "model_name": "gsdrunet",
            "sigma": sigma,
            "gmap_strategy": gmap,
            "unsharp_cfg": unsharp,
            "dither_cfg": dither,
            "mode": "2d",
        })

    # Error category 2: snraware_* in 3D mode (shape error, wrapper now fixed)
    for model, sigma, gmap, unsharp, dither in itertools.product(
        ["snraware_small", "snraware_medium", "snraware_large"],
        SIGMAS, GMAP_STRATEGIES, UNSHARP_CFGS, DITHER_CFGS
    ):
        tasks.append({
            "model_name": model,
            "sigma": sigma,
            "gmap_strategy": gmap,
            "unsharp_cfg": unsharp,
            "dither_cfg": dither,
            "mode": "3d",
        })

    # Error category 3: imt-mrd_* in 3D mode (TorchScript input mismatch, wrapper now fixed)
    for model, sigma, gmap, unsharp, dither in itertools.product(
        ["imt-mrd_complex", "imt-mrd_residual"],
        SIGMAS, GMAP_STRATEGIES, UNSHARP_CFGS, DITHER_CFGS
    ):
        tasks.append({
            "model_name": model,
            "sigma": sigma,
            "gmap_strategy": gmap,
            "unsharp_cfg": unsharp,
            "dither_cfg": dither,
            "mode": "3d",
        })

    return tasks


def _load_done_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            done.add(obj.get("_hash", ""))
        except json.JSONDecodeError:
            pass
    return done


def main() -> None:
    tasks = build_error_tasks()
    done = _load_done_hashes(OUT_PATH)
    pending = [t for t in tasks if _task_hash(t) not in done]

    print(f"Error-category tasks: {len(tasks)}, already done: {len(tasks) - len(pending)}, pending: {len(pending)}")

    completed = errors = 0
    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for task in pending:
            h = _task_hash(task)
            result = run_trial(task)
            result["_hash"] = h
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            completed += 1
            if result["error"] is not None:
                errors += 1
            psnr_str = f"{result['psnr']:.2f}" if result["psnr"] is not None else "None"
            print(
                f"[{completed:4d}/{len(pending)}] {task['model_name']:20s} "
                f"s={task['sigma']} mode={task['mode']:2s} "
                f"PSNR={psnr_str}  err={'YES' if result['error'] else 'no'}"
            )

    print(f"\nDone. {completed} trials, {errors} errors -> {OUT_PATH}")


if __name__ == "__main__":
    main()
