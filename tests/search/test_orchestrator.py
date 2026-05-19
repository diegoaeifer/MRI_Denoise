# MRI_Denoise/tests/search/test_orchestrator.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from search.orchestrator import run_search


def test_run_search_completes_at_least_one_trial(tmp_path):
    """30-second budget should complete >=1 trial using the CPU stub path."""
    out_file = tmp_path / "results.jsonl"
    run_search(
        duration_hours=30 / 3600,
        max_workers=1,
        out_path=out_file,
        task_seed=42,
    )
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1, f"Expected >=1 result lines, got {len(lines)}"


def test_result_lines_are_valid_json(tmp_path):
    import json
    out_file = tmp_path / "results.jsonl"
    run_search(
        duration_hours=30 / 3600,
        max_workers=1,
        out_path=out_file,
        task_seed=0,
    )
    for line in out_file.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        assert "psnr" in obj
        assert "sigma" in obj
        assert "model_name" in obj


def test_resume_skips_done_tasks(tmp_path):
    """Second run with same output file should not duplicate completed tasks."""
    import json
    out_file = tmp_path / "results.jsonl"
    # First run: complete some trials
    run_search(duration_hours=30 / 3600, max_workers=1, out_path=out_file, task_seed=0)
    count_after_first = len(out_file.read_text(encoding="utf-8").splitlines())
    assert count_after_first >= 1

    # Second run with same file: no new lines should match existing hashes
    run_search(duration_hours=30 / 3600, max_workers=1, out_path=out_file, task_seed=0)
    all_lines = out_file.read_text(encoding="utf-8").splitlines()
    # Extract hashes to detect duplicates
    hashes = [json.loads(l)["_hash"] for l in all_lines]
    assert len(hashes) == len(set(hashes)), "Duplicate task hashes found after resume"
