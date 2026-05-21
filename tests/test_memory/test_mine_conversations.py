import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/mine_conversations.py")

def test_script_exists():
    assert SCRIPT.exists(), f"{SCRIPT} not found"

def test_dry_run_exits_zero():
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry_run"],
        capture_output=True, text=True,
        cwd="C:/projetos/MRI_Denoise"
    )
    assert r.returncode == 0, r.stderr
    assert "JSONL" in r.stdout or "session" in r.stdout.lower()
