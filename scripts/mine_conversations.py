"""Mine all Claude Code JSONL sessions into MemPalace.

Usage:
    python scripts/mine_conversations.py [--dry_run] [--wing mri-denoise]

Notes:
    - Uses `mempalace --palace <dir> mine <projects_dir> --mode convos`
    - --palace is a global flag that must precede the subcommand
    - mine operates on a directory, not individual files
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

PROJECTS_DIR = Path(r"C:\Users\Pichau\.claude\projects")
PALACE_DIR   = Path(r"C:\projetos\.mempalace")
MEMPALACE    = Path(r"C:\projetos\mempalace-venv\Scripts\mempalace.exe")


def find_jsonl_files(projects_dir: Path) -> list[Path]:
    return sorted(projects_dir.rglob("*.jsonl"))


def mine_directory(projects_dir: Path, wing: str, dry_run: bool = False) -> int:
    """Mine the entire projects directory using convos mode."""
    cmd = [
        str(MEMPALACE),
        "--palace", str(PALACE_DIR),
        "mine", str(projects_dir),
        "--mode", "convos",
        "--wing", wing,
    ]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] mining failed:\n{result.stderr.strip()}")
    else:
        print(result.stdout)
    return result.returncode


def main() -> None:
    if not MEMPALACE.exists():
        print(f"[ERROR] mempalace executable not found: {MEMPALACE}")
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--wing", default="mri-denoise")
    p.add_argument("--projects_dir", default=str(PROJECTS_DIR))
    args = p.parse_args()

    projects_dir = Path(args.projects_dir)
    if not projects_dir.exists():
        print(f"[ERROR] projects_dir does not exist: {projects_dir}")
        sys.exit(1)

    sessions = find_jsonl_files(projects_dir)
    print(f"Found {len(sessions)} JSONL session(s)")

    if args.dry_run:
        print(f"\n[DRY_RUN] Would mine directory: {projects_dir}")
        rc = mine_directory(projects_dir, args.wing, dry_run=True)
        sys.exit(rc)

    rc = mine_directory(projects_dir, args.wing, dry_run=False)
    sys.exit(rc)


if __name__ == "__main__":
    main()
