"""Print a MemPalace wake-up summary for the start of a new session.

Usage:
    python scripts/memory_wake_up.py [--wing mri-denoise] [--query "current focus"]

Paste the output into the first message of a new Claude Code session to
inject prior context without reading old transcripts.
"""
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path

MEMPALACE  = Path(os.environ.get("MEMPALACE_EXE",  r"C:\projetos\mempalace-venv\Scripts\mempalace.exe"))
PALACE_DIR = Path(os.environ.get("MEMPALACE_DIR",  r"C:\projetos\.mempalace"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wing", default="mri-denoise")
    p.add_argument("--query", default="", help="Optional focus topic for the briefing")
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    # Prefer the wake-up subcommand (available in MemPalace >= 3.3.5)
    # which outputs L0 + L1 context (~600-900 tokens). Fall back to
    # search if it is unavailable.
    wake_cmd = [
        str(MEMPALACE), "--palace", str(PALACE_DIR),
        "wake-up",
    ]
    if args.wing:
        wake_cmd += ["--wing", args.wing]

    result = subprocess.run(wake_cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        # Fall back to search
        search_cmd = [
            str(MEMPALACE), "--palace", str(PALACE_DIR),
            "search", args.query or "project status recent work",
            "--results", str(args.top_k),
        ]
        if args.wing:
            search_cmd += ["--wing", args.wing]

        result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"[ERROR] {result.stderr.strip()}")
            return

    print("=" * 72)
    print("MEMORY WAKE-UP — paste this into your new session")
    print("=" * 72)
    print(result.stdout)


if __name__ == "__main__":
    main()
