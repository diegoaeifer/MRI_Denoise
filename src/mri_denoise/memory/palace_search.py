"""Thin wrapper around the MemPalace CLI for querying conversation memories.

Usage in scripts:
    from src.mri_denoise.memory.palace_search import search_memory
    results = search_memory("what optimizer did we use for NAFNet fine-tuning?")
    for r in results:
        print(r.score, r.content)

MemPalace v3.3.5 search output format (plain text, no JSON mode):
    [N] wing / room
        Source: <filename>
        Match:  cosine=X  bm25=Y

        <content block>

    ────────────────────────...
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MEMPALACE = Path(os.environ.get("MEMPALACE_EXE", r"C:\projetos\mempalace-venv\Scripts\mempalace.exe"))
PALACE_DIR = Path(os.environ.get("MEMPALACE_DIR", r"C:\projetos\.mempalace"))

# Matches the result header line: [N] wing / room
_HEADER_RE = re.compile(r"^\s*\[(\d+)\]\s+(\S+)\s*/\s*(\S+)")
# Matches: Match:  cosine=0.781  bm25=1.297
_MATCH_RE = re.compile(r"cosine=([\d.]+)")
# Separator line
_SEP_RE = re.compile(r"^[\s─\-]{20,}$")


@dataclass
class MemoryResult:
    content: str
    score: float       # cosine distance — lower = closer match (0.0 = exact duplicate)
    room: str
    wing: str


def _run_cli_search(query: str, top_k: int, wing: str | None) -> list[dict[str, Any]]:
    """Call the mempalace CLI and parse its plain-text output into dicts.

    Returns a list of dicts with keys: document, distance, metadata.
    Each metadata dict has 'room' and 'wing' keys.
    """
    cmd = [
        str(MEMPALACE),
        "--palace", str(PALACE_DIR),
        "search", query,
        "--results", str(top_k),
    ]
    if wing:
        cmd += ["--wing", wing]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )

    # Combine stdout and stderr — mempalace writes warnings to stderr but
    # results to stdout; stdout is what we parse.
    output = proc.stdout
    if not output.strip():
        return []

    return _parse_plain_text(output)


def _parse_plain_text(text: str) -> list[dict[str, Any]]:
    """Parse MemPalace v3.3.5 human-readable search output.

    Block structure per result:
        [N] wing / room           <- header
            Source: <file>
            Match:  cosine=X ...

            <content lines>       <- everything until separator or next header

        ──────────────────        <- separator
    """
    results: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    content_lines: list[str] = []

    def flush(item: dict[str, Any] | None, lines: list[str]) -> dict[str, Any] | None:
        if item is not None:
            item["document"] = "\n".join(lines).strip()
            results.append(item)
        return None

    for line in text.splitlines():
        header_m = _HEADER_RE.match(line)
        if header_m:
            current = flush(current, content_lines)
            content_lines = []
            current = {
                "distance": 1.0,
                "metadata": {
                    "wing": header_m.group(2),
                    "room": header_m.group(3),
                },
                "document": "",
            }
            continue

        if current is None:
            continue

        if _SEP_RE.match(line):
            continue

        match_m = _MATCH_RE.search(line)
        if match_m and "Source:" not in line:
            current["distance"] = float(match_m.group(1))
            continue

        if line.strip().startswith("Source:"):
            continue

        content_lines.append(line.rstrip())

    # Flush the last block — must be called after the loop so the final
    # parsed entry is not silently dropped.
    flush(current, content_lines)
    return results


def search_memory(
    query: str,
    top_k: int = 5,
    wing: str | None = None,
) -> list[MemoryResult]:
    """Search MemPalace for memories relevant to *query*.

    Args:
        query:  Natural language question or keyword phrase.
        top_k:  Maximum number of results to return.
        wing:   Restrict search to a specific Wing (e.g. "mri_denoise").

    Returns:
        List of MemoryResult sorted by relevance (best cosine score first).
    """
    raw = _run_cli_search(query, top_k, wing)
    results = []
    for item in raw:
        meta = item.get("metadata", {})
        results.append(MemoryResult(
            content=item.get("document", ""),
            score=float(item.get("distance", 1.0)),
            room=meta.get("room", ""),
            wing=meta.get("wing", ""),
        ))
    return sorted(results, key=lambda r: r.score)
