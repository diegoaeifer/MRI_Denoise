# MemPalace Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install MemPalace, register it as a Claude Code MCP server with auto-save hooks, mine all existing conversation JSONL history, and expose a Python search API usable from the research pipeline.

**Architecture:**
MemPalace runs as a local MCP server (`mempalace-mcp`) registered in Claude Code's global MCP settings. A Wings/Rooms hierarchy organizes memories by project (`mri-denoise`, `radiology-toolbox`). Auto-save hooks fire every 15 messages and before context compaction. The Python API (`Palace`, `Searcher`) is callable from research scripts to retrieve prior decisions, model choices, and dataset notes.

**Tech Stack:** Python 3.12, MemPalace (`pip install mempalace`), ChromaDB ≥1.5.4, SQLite (knowledge graph), Claude Code MCP protocol, PowerShell (Windows hook scripts).

**Conversation data location:** `C:\Users\Pichau\.claude\projects\C--projetos\` (JSONL files — one per session)

---

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `C:\Users\Pichau\.claude\settings.json` | **Modify** | Register `mempalace-mcp` in `mcpServers` |
| `C:\Users\Pichau\.claude\settings.local.json` | **Modify** | Add auto-save hooks (`SaveHook`, `PreCompactHook`) |
| `C:\projetos\mempalace-venv\` | **Create** | Isolated venv for MemPalace (keeps it out of project deps) |
| `C:\projetos\mempalace-init.py` | **Create** | One-shot: initialise palace, create Wings, mine history |
| `C:\projetos\MRI_Denoise\src\mri_denoise\memory\palace_search.py` | **Create** | Python API wrapper — `search_memory(query)` for use in scripts |
| `C:\projetos\MRI_Denoise\tests\test_memory\test_palace_search.py` | **Create** | Unit test for the search wrapper |
| `C:\projetos\MRI_Denoise\scripts\mine_conversations.py` | **Create** | CLI: scan JSONL dir → mine all sessions into MemPalace |

---

## Task 0: Install MemPalace in isolated venv

**Files:**
- Create: `C:\projetos\mempalace-venv\`

MemPalace needs ChromaDB ≥1.5.4 which may conflict with the project's pinned version. An isolated venv avoids dependency hell.

- [ ] **Step 1: Create the venv and install**

```powershell
# Run from PowerShell (not from the project venv)
python -m venv C:\projetos\mempalace-venv
C:\projetos\mempalace-venv\Scripts\pip install "mempalace[anthropic]"
```

Expected: pip resolves chromadb>=1.5.4, installs mempalace, mempalace-mcp entry point available.

- [ ] **Step 2: Verify both CLI entry points exist**

```powershell
C:\projetos\mempalace-venv\Scripts\mempalace --version
C:\projetos\mempalace-venv\Scripts\mempalace-mcp --help
```

Expected output (approximate):
```
mempalace 0.x.x
usage: mempalace-mcp ...
```

- [ ] **Step 3: Initialise the palace**

```powershell
$env:ANTHROPIC_API_KEY = "your-key-here"  # only needed for classify/sweep; can skip for now
C:\projetos\mempalace-venv\Scripts\mempalace init --palace-dir C:\projetos\.mempalace
```

Expected: creates `C:\projetos\.mempalace\` with ChromaDB and SQLite backing stores.

- [ ] **Step 4: Create Wings for our projects**

```powershell
C:\projetos\mempalace-venv\Scripts\mempalace wing create mri-denoise --description "MRI Denoising project — models, datasets, benchmarks, adapters"
C:\projetos\mempalace-venv\Scripts\mempalace wing create radiology-toolbox --description "Multi-project radiology AI workspace"
```

Expected: wings listed in `mempalace wing list`.

- [ ] **Step 5: Commit nothing — this is infrastructure, not code**

```
# No git commit for this task — mempalace-venv and .mempalace are gitignored
```

Add to `C:\projetos\MRI_Denoise\.gitignore` (if not already present):
```
/.mempalace/
/mempalace-venv/
```

---

## Task 1: Register MCP Server in Claude Code

**Files:**
- Modify: `C:\Users\Pichau\.claude\settings.json` (global Claude Code settings)

Claude Code discovers MCP servers from `mcpServers` in `~/.claude/settings.json`. Adding the `mempalace-mcp` entry makes 40+ memory tools available in every Claude Code session.

- [ ] **Step 1: Locate or create the global settings file**

```powershell
# Check if it exists
Test-Path "C:\Users\Pichau\.claude\settings.json"
# If not: New-Item "C:\Users\Pichau\.claude\settings.json" -ItemType File
```

- [ ] **Step 2: Add the mcpServers entry**

Open `C:\Users\Pichau\.claude\settings.json` and merge this JSON (preserve any existing keys):

```json
{
  "mcpServers": {
    "mempalace": {
      "command": "C:\\projetos\\mempalace-venv\\Scripts\\mempalace-mcp.exe",
      "args": ["--palace-dir", "C:\\projetos\\.mempalace"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

- [ ] **Step 3: Restart Claude Code and verify MCP is connected**

```
# In a new Claude Code session, type:
/mcp
```

Expected: `mempalace` listed as connected with status `ready`. You should see tools like `mempalace_search`, `mempalace_add_drawer` in the tool list.

- [ ] **Step 4: Quick smoke test from Claude Code**

In a Claude Code chat session, ask:
> "Use mempalace_search to find anything about NAFNet"

Expected: returns results (or "no results yet" — empty palace is fine here, we mine in Task 3).

---

## Task 2: Configure Auto-Save Hooks

**Files:**
- Modify: `C:\Users\Pichau\.claude\settings.local.json`

Two hooks fire automatically:
- **SaveHook**: every 15 messages → calls `mempalace sweep` on the current session JSONL
- **PreCompactHook**: before context compression → saves a summary drawer

- [ ] **Step 1: Write the save hook script**

Create `C:\projetos\.mempalace\hooks\save_hook.ps1`:

```powershell
# save_hook.ps1 — Called by Claude Code after every 15 messages
param([string]$SessionFile)

$mp = "C:\projetos\mempalace-venv\Scripts\mempalace.exe"
$palaceDir = "C:\projetos\.mempalace"

if (-not $SessionFile) {
    # Derive from most recently modified JSONL in the projects dir
    $SessionFile = Get-ChildItem "C:\Users\Pichau\.claude\projects" -Filter "*.jsonl" -Recurse |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}

if ($SessionFile -and (Test-Path $SessionFile)) {
    & $mp sweep $SessionFile --palace-dir $palaceDir --wing mri-denoise 2>&1 |
        Out-File "C:\projetos\.mempalace\hooks\save_hook.log" -Append
}
```

- [ ] **Step 2: Write the pre-compact hook script**

Create `C:\projetos\.mempalace\hooks\precompact_hook.ps1`:

```powershell
# precompact_hook.ps1 — Called before Claude Code compresses context
param([string]$SessionFile)

$mp = "C:\projetos\mempalace-venv\Scripts\mempalace.exe"
$palaceDir = "C:\projetos\.mempalace"

if (-not $SessionFile) {
    $SessionFile = Get-ChildItem "C:\Users\Pichau\.claude\projects" -Filter "*.jsonl" -Recurse |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}

if ($SessionFile -and (Test-Path $SessionFile)) {
    & $mp mine $SessionFile --palace-dir $palaceDir --wing mri-denoise --rooms auto 2>&1 |
        Out-File "C:\projetos\.mempalace\hooks\precompact_hook.log" -Append
}
```

- [ ] **Step 3: Register hooks in settings.local.json**

Edit `C:\Users\Pichau\.claude\settings.local.json` to add hooks alongside existing `permissions`:

```json
{
  "permissions": {
    "allow": [
      "Bash(mkdir -p \"C:/projetos/interpolation-mri/.agents\")",
      "Bash(cp \"C:/projetos/.agents/AGENTS.md\" \"C:/projetos/interpolation-mri/.agents/AGENTS.md\")"
    ]
  },
  "hooks": {
    "SaveHook": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "powershell -NonInteractive -File C:\\projetos\\.mempalace\\hooks\\save_hook.ps1"
          }
        ]
      }
    ],
    "PreCompactHook": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "powershell -NonInteractive -File C:\\projetos\\.mempalace\\hooks\\precompact_hook.ps1"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 4: Test the save hook manually**

```powershell
# Run the hook manually against the current session file
powershell -NonInteractive -File "C:\projetos\.mempalace\hooks\save_hook.ps1"
# Check log
Get-Content "C:\projetos\.mempalace\hooks\save_hook.log" -Tail 20
```

Expected: no errors; log shows rooms created or drawers added.

---

## Task 3: Mine Existing Conversation History

**Files:**
- Create: `C:\projetos\MRI_Denoise\scripts\mine_conversations.py`

The current session JSONL is at `C:\Users\Pichau\.claude\projects\C--projetos\46ae2736-61f0-4601-a5a2-1ac4a87518d8.jsonl` and there may be multiple sessions. This script mines all of them.

- [ ] **Step 1: Write the failing test**

Create `C:\projetos\MRI_Denoise\tests\test_memory\test_mine_conversations.py`:

```python
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
```

- [ ] **Step 2: Run test to confirm failure**

```powershell
Set-Location C:\projetos\MRI_Denoise
python -m pytest tests/test_memory/test_mine_conversations.py -v
```

Expected: `FAILED` — `scripts/mine_conversations.py` does not exist yet.

- [ ] **Step 3: Implement `scripts/mine_conversations.py`**

```python
"""Mine all Claude Code JSONL sessions into MemPalace.

Usage:
    python scripts/mine_conversations.py [--dry_run] [--wing mri-denoise]
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


def mine_session(jsonl: Path, wing: str, dry_run: bool) -> int:
    if dry_run:
        print(f"  [DRY_RUN] Would mine: {jsonl}")
        return 0
    cmd = [
        str(MEMPALACE), "sweep", str(jsonl),
        "--palace-dir", str(PALACE_DIR),
        "--wing", wing,
        "--rooms", "auto",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {jsonl.name}: {result.stderr.strip()}")
    else:
        print(f"  [OK] {jsonl.name}: {result.stdout.strip()[:120]}")
    return result.returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--wing", default="mri-denoise")
    p.add_argument("--projects_dir", default=str(PROJECTS_DIR))
    args = p.parse_args()

    sessions = find_jsonl_files(Path(args.projects_dir))
    print(f"Found {len(sessions)} JSONL session(s)")
    for s in sessions:
        print(f"  {s}")

    if args.dry_run:
        print("\n[DRY_RUN] No mining performed.")
        return

    errors = 0
    for s in sessions:
        errors += mine_session(s, args.wing, dry_run=False)

    print(f"\nDone. {len(sessions) - errors}/{len(sessions)} sessions mined successfully.")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
python -m pytest tests/test_memory/test_mine_conversations.py -v
```

Expected: `PASSED` for both tests.

- [ ] **Step 5: Run the actual mining (dry run first)**

```powershell
python scripts/mine_conversations.py --dry_run
```

Expected: lists all `.jsonl` files found under `C:\Users\Pichau\.claude\projects\`.

- [ ] **Step 6: Mine for real**

```powershell
python scripts/mine_conversations.py --wing mri-denoise
```

Expected: each session logged as `[OK]`. After completion, verify with:

```powershell
C:\projetos\mempalace-venv\Scripts\mempalace search "NAFNet denoising" --palace-dir C:\projetos\.mempalace
```

Expected: returns drawer(s) referencing NAFNet fine-tuning, LoRA adapters, benchmark results from our conversations.

- [ ] **Step 7: Commit**

```powershell
git add scripts/mine_conversations.py tests/test_memory/test_mine_conversations.py
git commit -m "feat: add conversation mining script for MemPalace"
```

---

## Task 4: Python Search API Wrapper

**Files:**
- Create: `C:\projetos\MRI_Denoise\src\mri_denoise\memory\__init__.py`
- Create: `C:\projetos\MRI_Denoise\src\mri_denoise\memory\palace_search.py`
- Create: `C:\projetos\MRI_Denoise\tests\test_memory\test_palace_search.py`

This makes MemPalace searchable from Python scripts (e.g., retrieve prior decisions before a training run).

- [ ] **Step 1: Write the failing test**

Create `C:\projetos\MRI_Denoise\tests\test_memory\test_palace_search.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from src.mri_denoise.memory.palace_search import search_memory, MemoryResult


def test_memory_result_dataclass():
    r = MemoryResult(content="test content", score=0.9, room="models", wing="mri-denoise")
    assert r.content == "test content"
    assert r.score == 0.9


def test_search_memory_returns_list():
    fake_results = [
        {"document": "NAFNet was trained with lr=5e-4", "distance": 0.1, "metadata": {"room": "models", "wing": "mri-denoise"}},
    ]
    with patch("src.mri_denoise.memory.palace_search._run_cli_search", return_value=fake_results):
        results = search_memory("NAFNet learning rate")
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MemoryResult)
    assert "NAFNet" in results[0].content


def test_search_memory_empty_palace():
    with patch("src.mri_denoise.memory.palace_search._run_cli_search", return_value=[]):
        results = search_memory("nonexistent topic xyz")
    assert results == []


def test_search_memory_top_k():
    fake_results = [
        {"document": f"result {i}", "distance": float(i) * 0.1, "metadata": {"room": "r", "wing": "w"}}
        for i in range(10)
    ]
    with patch("src.mri_denoise.memory.palace_search._run_cli_search", return_value=fake_results[:3]):
        results = search_memory("query", top_k=3)
    assert len(results) <= 3
```

- [ ] **Step 2: Run test to confirm failure**

```powershell
python -m pytest tests/test_memory/test_palace_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.mri_denoise.memory'`

- [ ] **Step 3: Create `src/mri_denoise/memory/__init__.py`**

```python
# empty
```

- [ ] **Step 4: Implement `src/mri_denoise/memory/palace_search.py`**

```python
"""Thin wrapper around the MemPalace CLI for querying conversation memories.

Usage in scripts:
    from src.mri_denoise.memory.palace_search import search_memory
    results = search_memory("what optimizer did we use for NAFNet fine-tuning?")
    for r in results:
        print(r.score, r.content)
"""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MEMPALACE = Path(r"C:\projetos\mempalace-venv\Scripts\mempalace.exe")
PALACE_DIR = Path(r"C:\projetos\.mempalace")


@dataclass
class MemoryResult:
    content: str
    score: float       # lower distance = better match (0.0 = exact)
    room: str
    wing: str


def _run_cli_search(query: str, top_k: int, wing: str | None) -> list[dict[str, Any]]:
    cmd = [
        str(MEMPALACE), "search", query,
        "--palace-dir", str(PALACE_DIR),
        "--top-k", str(top_k),
        "--output", "json",
    ]
    if wing:
        cmd += ["--wing", wing]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def search_memory(
    query: str,
    top_k: int = 5,
    wing: str | None = None,
) -> list[MemoryResult]:
    """Search MemPalace for memories relevant to `query`.

    Args:
        query: Natural language question or keyword phrase.
        top_k: Maximum number of results to return.
        wing: Restrict search to a specific Wing (e.g. "mri-denoise").

    Returns:
        List of MemoryResult sorted by relevance (best first).
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
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
python -m pytest tests/test_memory/test_palace_search.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/mri_denoise/memory/ tests/test_memory/test_palace_search.py
git commit -m "feat: add MemPalace Python search API wrapper"
```

---

## Task 5: Wake-Up Integration (Context Injection at Session Start)

**Files:**
- Create: `C:\projetos\MRI_Denoise\scripts\memory_wake_up.py`

The `mempalace wake-up` command generates a context summary for the start of a new Claude Code session. This script wraps it so you can run it from the project and get a focused briefing.

- [ ] **Step 1: Implement `scripts/memory_wake_up.py`**

```python
"""Print a MemPalace wake-up summary for the start of a new session.

Usage:
    python scripts/memory_wake_up.py [--wing mri-denoise] [--query "current focus"]
    
Paste the output into the first message of a new Claude Code session to
inject prior context without reading old transcripts.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

MEMPALACE  = Path(r"C:\projetos\mempalace-venv\Scripts\mempalace.exe")
PALACE_DIR = Path(r"C:\projetos\.mempalace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wing", default="mri-denoise")
    p.add_argument("--query", default="", help="Optional focus topic for the briefing")
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    cmd = [
        str(MEMPALACE), "wake-up",
        "--palace-dir", str(PALACE_DIR),
        "--wing", args.wing,
        "--top-k", str(args.top_k),
    ]
    if args.query:
        cmd += ["--query", args.query]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr.strip()}")
        return

    print("=" * 72)
    print("MEMORY WAKE-UP — paste this into your new session")
    print("=" * 72)
    print(result.stdout)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify output**

```powershell
python scripts/memory_wake_up.py --wing mri-denoise --query "denoising model training"
```

Expected: a Markdown-formatted briefing listing recent decisions, model names, dataset notes, and open questions extracted from mined conversation history.

- [ ] **Step 3: Commit**

```powershell
git add scripts/memory_wake_up.py
git commit -m "feat: add MemPalace wake-up session briefing script"
```

---

## Verification — End-to-End Checklist

```powershell
# 1. Verify installation
C:\projetos\mempalace-venv\Scripts\mempalace --version

# 2. Verify MCP is visible in Claude Code
# Open Claude Code, run: /mcp
# Expect: mempalace listed as connected

# 3. Verify mining worked
C:\projetos\mempalace-venv\Scripts\mempalace search "NAFNet LoRA fine-tuning" --palace-dir C:\projetos\.mempalace
# Expect: results referencing our LoRA/FouRA adapter work

# 4. Verify Python API
python -c "
from src.mri_denoise.memory.palace_search import search_memory
results = search_memory('NAFNet denoising benchmark', wing='mri-denoise')
print(f'{len(results)} results')
for r in results[:3]:
    print(f'  [{r.score:.3f}] {r.content[:80]}')
"

# 5. Verify wake-up
python scripts/memory_wake_up.py --query "current project status"
# Expect: formatted briefing from memory

# 6. Run full test suite
python -m pytest tests/test_memory/ -v
# Expect: all tests pass
```

---

## Notes / Risks

- **ANTHROPIC_API_KEY required for `sweep` / `classify`**: The `mine` command without classification works without the key; `sweep` (which auto-classifies into rooms) needs it. Set in environment before running Task 3.
- **ChromaDB version conflict**: If the project venv already has ChromaDB <1.5.4, MemPalace must stay in its own venv. The `palace_search.py` wrapper calls the CLI (not the Python API) to avoid import conflicts.
- **Windows PowerShell hooks**: Claude Code on Windows may not support `SaveHook`/`PreCompactHook` yet (as of mid-2026). If hooks don't fire, run `mine_conversations.py` manually after each session instead.
- **JSONL path**: The session JSONL path (`46ae2736-...jsonl`) changes per session. `mine_conversations.py` uses `rglob("*.jsonl")` to catch all sessions automatically.
- **Privacy**: MemPalace stores all conversation content locally at `C:\projetos\.mempalace\`. Add `.mempalace/` to `.gitignore` — never commit it.
- **Palace directory**: Use `C:\projetos\.mempalace\` (workspace-level) not inside `MRI_Denoise/` — it covers all projects in the workspace.
