# MRI_Denoise/scripts/search/orchestrator.py
"""8-hour parallel parameter search orchestrator.

Usage:
    python -m scripts.search.orchestrator
    python -m scripts.search.orchestrator --hours 8 --workers 2 --out results.jsonl

Results are appended live to JSONL. Safe to interrupt and resume — completed
tasks are de-duplicated by MD5 hash of the task config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from search.task_grid import build_task_grid
from search.eval_pipeline import run_trial

OUT_DEFAULT = Path("C:/projetos/benchmark_results/search/results.jsonl")


def _task_hash(task: dict) -> str:
    key = json.dumps(task, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


def _load_done_hashes(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            done.add(obj.get("_hash", ""))
        except json.JSONDecodeError:
            pass
    return done


def run_search(
    duration_hours: float = 8.0,
    max_workers: int = 2,
    out_path: Path = OUT_DEFAULT,
    task_seed: int = 0,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + duration_hours * 3600
    tasks = build_task_grid(seed=task_seed)
    done_hashes = _load_done_hashes(out_path)

    pending = [t for t in tasks if _task_hash(t) not in done_hashes]
    print(
        f"[orchestrator] {len(tasks)} total tasks, {len(pending)} pending, "
        f"budget={duration_hours:.3f}h, workers={max_workers}"
    )

    completed = 0
    errors = 0
    task_iter = iter(pending)

    with ProcessPoolExecutor(max_workers=max_workers) as ex, \
         open(out_path, "a", encoding="utf-8") as fout:

        futures: dict = {}

        def _submit_next() -> bool:
            if time.monotonic() >= deadline:
                return False
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            h = _task_hash(task)
            fut = ex.submit(run_trial, task)
            futures[fut] = (task, h)
            return True

        # Fill worker pool
        for _ in range(max_workers):
            if not _submit_next():
                break

        for fut in as_completed(futures):
            task, h = futures.pop(fut)
            try:
                result = fut.result()
            except Exception as exc:
                result = dict(task, psnr=None, ssim=None,
                              error=str(exc), elapsed_s=0.0)
                errors += 1

            result["_hash"] = h
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            completed += 1

            psnr_str = f"{result['psnr']:.2f}" if result["psnr"] is not None else "None"
            print(
                f"[{completed:5d}] {task['model_name']:20s} sigma={task['sigma']} "
                f"gmap={task['gmap_strategy']:12s} "
                f"unsharp={task['unsharp_cfg']['name']:8s} "
                f"PSNR={psnr_str} dB  mode={task['mode']}  err={errors}"
            )

            _submit_next()

    print(f"[orchestrator] Done. {completed} trials, {errors} errors -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",   type=float, default=8.0)
    parser.add_argument("--workers", type=int,   default=2)
    parser.add_argument("--out",     type=str,   default=str(OUT_DEFAULT))
    parser.add_argument("--seed",    type=int,   default=0)
    args = parser.parse_args()
    run_search(
        duration_hours=args.hours,
        max_workers=args.workers,
        out_path=Path(args.out),
        task_seed=args.seed,
    )
