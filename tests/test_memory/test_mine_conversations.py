import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]  # tests/test_memory/ -> tests/ -> repo root
SCRIPT = REPO_ROOT / "scripts/mine_conversations.py"


def test_script_exists():
    assert SCRIPT.exists(), f"{SCRIPT} not found"


def test_dry_run_exits_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry_run"],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT)
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "Found" in r.stdout, f"Expected 'Found' in output, got: {r.stdout!r}"
    assert "[DRY_RUN]" in r.stdout, f"Expected '[DRY_RUN]' in output, got: {r.stdout!r}"


def test_bad_projects_dir_exits_nonzero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--projects_dir", "C:/nonexistent/path/xyz123"],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT)
    )
    assert r.returncode != 0, "Expected nonzero exit for missing projects_dir"
