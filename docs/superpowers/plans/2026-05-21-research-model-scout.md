# Research Model Scout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse the `research-organized/` literature output to rank the most promising MRI denoising and super-resolution models, then search the internet in parallel to find pretrained weights for each.

**Architecture:**
`scripts/research_analyzer.py` walks all `.md` files in `C:\projetos\research-organized\`, extracts structured metadata (title, task, approach, promise score, key method), filters to MRI denoising/SR, deduplicates, scores, and writes `artifacts/promising_models.json`. The implementer of Task 1 then reads that JSON and spawns parallel WebSearch agents — one per top model — to find GitHub repos, HuggingFace pages, and pretrained weight files. `scripts/generate_pretrained_report.py` reads both JSON outputs and renders `artifacts/pretrained_models_report.md`.

**Tech Stack:** Python 3.14, pathlib, re, json — zero extra deps. Research data at `C:\projetos\research-organized\` (175 `.md` files, created by `research_pipeline.py`). Artifacts land in `C:\projetos\MRI_Denoise\artifacts\`.

---

## Context

### Research output format

Every `.md` file in `research-organized/` was produced by `research_pipeline.py` and has this exact structure:

```markdown
# Model Title or Acronym: Full Paper Name

**Source:** C:\projetos\research-inbox\...
**Modality:** MRI | **Task:** Denoising | **Approach:** Hybrid
**Promise:** 4/5

## Summary
Two-sentence summary...

**Key method:** short name of the technique
```

Fields of interest:
- **Modality** — `MRI`, `Multi`, `CT`, `XRay`, etc. We want `MRI` or `Multi`.
- **Task** — `Denoising`, `Super-Resolution`, `Enhancement`, `Reconstruction`, `Other`, `Segmentation`, etc.
  The folder path also encodes task (e.g., `MRI/Denoising/`, `MRI/Super-Resolution/`).
- **Promise** — 1–5 integer (5 = most promising).
- **Approach** — `DL`, `Deep Learning`, `Hybrid`, `Classical`, `FoundationModel`, etc.
- **Key method** — short description, often the model name or technique.

### Scoring rationale

`score = promise + approach_bonus + github_bonus + task_bonus`

- `approach_bonus`: +1 for Hybrid, +2 for FoundationModel or Deep Learning (prefer learnable models)
- `github_bonus`: +2 if a GitHub URL appears anywhere in the file (code available → pretrained more likely)
- `task_bonus`: +1 for Denoising, +0.5 for Super-Resolution/Enhancement (directly relevant to our pipeline)

### Critical paths

| File | Action | Purpose |
|------|--------|---------|
| `scripts/research_analyzer.py` | **Create** | Parse research output → ranked JSON |
| `scripts/generate_pretrained_report.py` | **Create** | JSON → markdown report |
| `tests/test_research_analyzer.py` | **Create** | TDD for parser and scorer |
| `tests/test_generate_report.py` | **Create** | TDD for report generator |
| `artifacts/promising_models.json` | **Generated** | Ranked model list (top 25) |
| `artifacts/pretrained_search_results.json` | **Generated** | Web search results per model |
| `artifacts/pretrained_models_report.md` | **Generated** | Final human-readable report |

---

## Task 0: Research Analyzer Script

**Files:**
- Create: `scripts/research_analyzer.py`
- Create: `tests/test_research_analyzer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_research_analyzer.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO))

from scripts.research_analyzer import (
    extract_model_name,
    is_relevant,
    parse_md_file,
    score_model,
)

SAMPLE_MD_DENOISING = """\
# GAMBAS: Generalised-Hilbert Mamba for Super-resolution of Paediatric MRI

**Source:** C:\\projetos\\research-inbox\\135_GAMBAS.pdf
**Modality:** MRI | **Task:** Super-Resolution | **Approach:** Hybrid
**Promise:** 4/5

## Summary
GAMBAS is a hybrid CNN-Mamba model for ultra-low-field MRI super-resolution.

**Key method:** Hilbert curve 3D-to-1D Mamba serialization
"""

SAMPLE_MD_CT = """\
# Some CT Denoising Paper

**Source:** C:\\path\\ct_paper.pdf
**Modality:** CT | **Task:** Denoising | **Approach:** DL
**Promise:** 5/5

## Summary
CT denoising with a deep network.

**Key method:** DnCNN-CT
"""

SAMPLE_MD_SEG = """\
# Brain Segmentation with UNet

**Source:** C:\\path\\seg.pdf
**Modality:** MRI | **Task:** Segmentation | **Approach:** DL
**Promise:** 4/5

## Summary
Segmentation.

**Key method:** UNet
"""


def test_parse_md_fields(tmp_path):
    f = tmp_path / "test.md"
    f.write_text(SAMPLE_MD_DENOISING, encoding="utf-8")
    meta = parse_md_file(f)
    assert meta is not None
    assert "GAMBAS" in meta["title"]
    assert meta["modality"] == "MRI"
    assert meta["task"] == "Super-Resolution"
    assert meta["approach"] == "Hybrid"
    assert meta["promise"] == 4
    assert "Hilbert" in meta["key_method"]


def test_parse_md_no_github_when_absent(tmp_path):
    f = tmp_path / "test.md"
    f.write_text(SAMPLE_MD_DENOISING, encoding="utf-8")
    meta = parse_md_file(f)
    assert "github" not in meta


def test_parse_md_extracts_github(tmp_path):
    text = SAMPLE_MD_DENOISING + "\nhttps://github.com/author/GAMBAS\n"
    f = tmp_path / "test.md"
    f.write_text(text, encoding="utf-8")
    meta = parse_md_file(f)
    assert meta.get("github") == "https://github.com/author/GAMBAS"


def test_is_relevant_denoising():
    assert is_relevant({"modality": "MRI", "task": "Denoising"}, Path("MRI/Denoising/x.md")) is True


def test_is_relevant_sr_in_path():
    # task field says Other but folder says Super-Resolution
    assert is_relevant({"modality": "MRI", "task": "Other"}, Path("MRI/Super-Resolution/x.md")) is True


def test_is_relevant_ct_excluded():
    assert is_relevant({"modality": "CT", "task": "Denoising"}, Path("CT/Denoising/x.md")) is False


def test_is_relevant_mri_segmentation_excluded():
    assert is_relevant({"modality": "MRI", "task": "Segmentation"}, Path("MRI/Segmentation/x.md")) is False


def test_is_relevant_multi_denoising_included():
    assert is_relevant({"modality": "Multi", "task": "Denoising"}, Path("Multi/Denoising/x.md")) is True


def test_score_model_promise_is_dominant():
    low = {"promise": 2, "approach": "Classical", "task": "Denoising"}
    high = {"promise": 5, "approach": "Classical", "task": "Denoising"}
    assert score_model(high) > score_model(low)


def test_score_model_github_bonus():
    without = {"promise": 3, "approach": "Hybrid", "task": "Denoising"}
    with_gh = {"promise": 3, "approach": "Hybrid", "task": "Denoising", "github": "https://github.com/x/y"}
    assert score_model(with_gh) > score_model(without)


def test_score_model_dl_approach_bonus():
    classical = {"promise": 3, "approach": "Classical", "task": "Denoising"}
    dl = {"promise": 3, "approach": "Deep Learning", "task": "Denoising"}
    assert score_model(dl) > score_model(classical)


def test_extract_model_name_acronym():
    meta = {"title": "GAMBAS: Something Long Here", "key_method": "Hilbert mamba"}
    assert extract_model_name(meta) == "GAMBAS"


def test_extract_model_name_falls_back_to_key_method():
    meta = {"title": "a lower case title without acronym", "key_method": "Noise2Noise framework"}
    name = extract_model_name(meta)
    assert "Noise2Noise" in name


def test_main_script_runs_and_creates_json():
    r = subprocess.run(
        [sys.executable, "scripts/research_analyzer.py"],
        capture_output=True, text=True,
        cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr
    out = REPO / "artifacts" / "promising_models.json"
    assert out.exists(), "promising_models.json not created"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "top_models" in data
    assert len(data["top_models"]) > 0
    first = data["top_models"][0]
    assert "model_name" in first
    assert "score" in first
    assert "promise" in first
```

- [ ] **Step 2: Run test to confirm failure**

```powershell
Set-Location C:\projetos\MRI_Denoise
python -m pytest tests/test_research_analyzer.py -v
```

Expected: `ImportError` — `scripts/research_analyzer.py` does not exist yet.

- [ ] **Step 3: Implement `scripts/research_analyzer.py`**

```python
"""Parse research-organized/ and rank promising MRI denoising/SR models.

Usage:
    python scripts/research_analyzer.py [--research_dir PATH] [--top N]

Reads all .md files under C:/projetos/research-organized/ (or --research_dir),
filters to MRI/Multi modality + denoising/SR/enhancement tasks,
deduplicates by title, scores, and writes artifacts/promising_models.json.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

RESEARCH_DIR = Path(os.environ.get("RESEARCH_DIR", r"C:\projetos\research-organized"))
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

_TARGET_MODALITIES = {"MRI", "Multi"}
_TASK_KEYWORDS = {"denois", "super-resol", "superresol", "enhancement", "upscal", "reconstruct"}
_DL_APPROACHES = {"DL", "Deep Learning", "Hybrid", "FoundationModel", "Foundation Model"}


def parse_md_file(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    meta: dict = {}
    lines = text.splitlines()

    for line in lines:
        if line.startswith("# ") and "title" not in meta:
            meta["title"] = line[2:].strip()
            continue

        m = re.match(r"\*\*Modality:\*\*\s*([^|]+?)(?:\s*\|.*)?$", line)
        if m:
            meta["modality"] = m.group(1).strip()

        m = re.match(r".*\*\*Task:\*\*\s*([^|]+?)(?:\s*\|.*)?$", line)
        if m:
            meta["task"] = m.group(1).strip()

        m = re.match(r".*\*\*Approach:\*\*\s*(.+?)$", line)
        if m:
            meta["approach"] = m.group(1).strip()

        m = re.match(r"\*\*Promise:\*\*\s*(\d)/5", line)
        if m:
            meta["promise"] = int(m.group(1))

        m = re.match(r"\*\*Key method:\*\*\s*(.+?)$", line)
        if m:
            meta["key_method"] = m.group(1).strip()

        m = re.match(r"\*\*Source:\*\*\s*(.+?)$", line)
        if m:
            meta["source"] = m.group(1).strip()

    # Extract summary block (between "## Summary" and "**Key method:**")
    summary_match = re.search(r"## Summary\s*\n(.*?)(?=\n\*\*Key method|\Z)", text, re.DOTALL)
    if summary_match:
        meta["summary"] = summary_match.group(1).strip()

    # Extract GitHub URL from full text
    gh = re.search(r"https?://github\.com/[\w\-./]+", text)
    if gh:
        meta["github"] = gh.group(0).rstrip(".,)")

    meta["file"] = str(path)
    return meta if "title" in meta else None


def is_relevant(meta: dict, path: Path) -> bool:
    if meta.get("modality", "") not in _TARGET_MODALITIES:
        return False
    task = meta.get("task", "").lower()
    path_lower = str(path).lower()
    return any(kw in task or kw in path_lower for kw in _TASK_KEYWORDS)


def score_model(meta: dict) -> float:
    s = float(meta.get("promise", 0))
    approach = meta.get("approach", "")
    if any(a in approach for a in _DL_APPROACHES):
        s += 1 if "Hybrid" in approach else 2
    if "github" in meta:
        s += 2
    task = meta.get("task", "").lower()
    if "denois" in task:
        s += 1
    elif any(kw in task for kw in {"super", "resol", "upscal"}):
        s += 0.5
    return s


def extract_model_name(meta: dict) -> str:
    title = meta.get("title", "")
    # Prefer leading acronym (ALL-CAPS or CamelCase word before colon/comma)
    m = re.match(r"^([A-Z][A-Za-z0-9\-]{2,})[:\s,;]", title)
    if m:
        return m.group(1)
    key = meta.get("key_method", "")
    return (key or title)[:60]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--research_dir", default=str(RESEARCH_DIR))
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    research_dir = Path(args.research_dir)
    seen: set[str] = set()
    models: list[dict] = []

    for md_path in research_dir.rglob("*.md"):
        if md_path.name in {"INDEX.md"}:
            continue
        meta = parse_md_file(md_path)
        if meta is None:
            continue
        if not is_relevant(meta, md_path):
            continue
        title_key = meta.get("title", "").lower()[:50]
        if title_key in seen:
            continue
        seen.add(title_key)
        meta["model_name"] = extract_model_name(meta)
        meta["score"] = score_model(meta)
        models.append(meta)

    ranked = sorted(models, key=lambda x: x["score"], reverse=True)
    top = ranked[: args.top]

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    output = {
        "total_found": len(ranked),
        "top_models": top,
    }
    out_path = ARTIFACTS_DIR / "promising_models.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Found {len(ranked)} relevant models (MRI/Multi denoising/SR/enhancement)")
    print(f"\nTop {min(10, len(top))} by score:")
    for m in top[:10]:
        github_tag = " [GitHub]" if "github" in m else ""
        print(f"  [{m['score']:.1f}] {m['model_name']:<30} | {m.get('task','?'):<20} | Promise {m.get('promise','?')}/5{github_tag}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_research_analyzer.py -v
```

Expected: all tests pass. The `test_main_script_runs_and_creates_json` test also runs the script and verifies `artifacts/promising_models.json` is created.

- [ ] **Step 5: Inspect output**

```powershell
python scripts/research_analyzer.py
```

Check output looks reasonable — top models should be recognizable MRI denoising/SR techniques with promise ≥ 3. Verify `artifacts/promising_models.json` has 25+ entries.

- [ ] **Step 6: Commit**

```powershell
git add scripts/research_analyzer.py tests/test_research_analyzer.py artifacts/promising_models.json
git commit -m "feat: add research analyzer to rank promising MRI denoising/SR models"
```

---

## Task 1: Parallel Web Search for Pretrained Models

**Files:**
- Generate: `artifacts/pretrained_search_results.json` (written by this task's implementer via WebSearch)

This task is an **agentic research task**, not a Python implementation task. The implementer reads `artifacts/promising_models.json`, then uses WebSearch to search for each top model and writes structured results.

**Implementer instructions:**

1. Read `C:\projetos\MRI_Denoise\artifacts\promising_models.json`
2. Take the top 15 models (by score) from `top_models`
3. For each model, search for ALL of the following:
   - `{model_name} pretrained weights download MRI`
   - `{model_name} site:github.com`
   - `{model_name} site:huggingface.co`
   - `{model_name} site:paperswithcode.com`
4. Write results to `C:\projetos\MRI_Denoise\artifacts\pretrained_search_results.json` in this exact format:

```json
{
  "ModelName": {
    "searched_for": "ModelName pretrained weights MRI",
    "github": "https://github.com/author/repo",
    "huggingface": "https://huggingface.co/org/model",
    "papers_with_code": "https://paperswithcode.com/paper/...",
    "pretrained_available": true,
    "pretrained_notes": "Weights at releases page / HuggingFace / Google Drive",
    "license": "MIT",
    "search_date": "2026-05-21"
  }
}
```

- Set `pretrained_available` to `true` if you find a clear download link for weights; `false` if code exists but no weights; `null` if no code found at all.
- Set fields to `null` if not found.

- [ ] **Step 1: Read `artifacts/promising_models.json`** and extract `model_name` for top 15 entries.

- [ ] **Step 2: Search for each model in parallel** using WebSearch (fire all searches, then collect results).

Search each model with all 4 query patterns above.

- [ ] **Step 3: Write `artifacts/pretrained_search_results.json`** with the collected results.

- [ ] **Step 4: Commit**

```powershell
git add artifacts/pretrained_search_results.json
git commit -m "feat: add pretrained model availability search results"
```

---

## Task 2: Report Generation Script

**Files:**
- Create: `scripts/generate_pretrained_report.py`
- Create: `tests/test_generate_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate_report.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def fake_artifacts(tmp_path, monkeypatch):
    """Provide minimal JSON fixtures so the script runs without real artifacts."""
    # Patch artifact path via env var
    art = REPO / "artifacts"
    art.mkdir(exist_ok=True)

    models_json = {
        "total_found": 3,
        "top_models": [
            {
                "model_name": "GAMBAS",
                "title": "GAMBAS: Super-resolution of Paediatric MRI",
                "task": "Super-Resolution",
                "approach": "Hybrid",
                "promise": 4,
                "score": 7.5,
                "key_method": "Hilbert Mamba",
                "summary": "SR model for ultra-low-field MRI.",
                "github": "https://github.com/fake/gambas",
            },
            {
                "model_name": "Noise2Average",
                "title": "Noise2Average: denoising without ground truth",
                "task": "Denoising",
                "approach": "Deep Learning",
                "promise": 4,
                "score": 7.0,
                "key_method": "Iterative residual learning",
                "summary": "Self-supervised MRI denoising.",
            },
        ],
    }
    search_json = {
        "GAMBAS": {
            "github": "https://github.com/fake/gambas",
            "huggingface": None,
            "papers_with_code": None,
            "pretrained_available": True,
            "pretrained_notes": "Weights at releases page",
            "license": "MIT",
            "search_date": "2026-05-21",
        },
        "Noise2Average": {
            "github": None,
            "huggingface": None,
            "papers_with_code": None,
            "pretrained_available": False,
            "pretrained_notes": None,
            "license": None,
            "search_date": "2026-05-21",
        },
    }
    (art / "promising_models.json").write_text(json.dumps(models_json), encoding="utf-8")
    (art / "pretrained_search_results.json").write_text(json.dumps(search_json), encoding="utf-8")


def test_generate_report_creates_md():
    r = subprocess.run(
        [sys.executable, "scripts/generate_pretrained_report.py"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr
    report = REPO / "artifacts" / "pretrained_models_report.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "GAMBAS" in text
    assert "Noise2Average" in text


def test_report_contains_table():
    r = subprocess.run(
        [sys.executable, "scripts/generate_pretrained_report.py"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr
    text = (REPO / "artifacts" / "pretrained_models_report.md").read_text()
    assert "| Rank |" in text
    assert "| 1 |" in text


def test_report_pretrained_checkmark():
    r = subprocess.run(
        [sys.executable, "scripts/generate_pretrained_report.py"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr
    text = (REPO / "artifacts" / "pretrained_models_report.md").read_text()
    assert "✅" in text   # GAMBAS has pretrained_available=True
    assert "❌" in text   # Noise2Average has pretrained_available=False


def test_report_contains_detail_sections():
    r = subprocess.run(
        [sys.executable, "scripts/generate_pretrained_report.py"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stderr
    text = (REPO / "artifacts" / "pretrained_models_report.md").read_text()
    assert "### GAMBAS" in text
    assert "### Noise2Average" in text
```

- [ ] **Step 2: Run test to confirm failure**

```powershell
python -m pytest tests/test_generate_report.py -v
```

Expected: `ImportError` or `FileNotFoundError` — `scripts/generate_pretrained_report.py` does not exist.

- [ ] **Step 3: Implement `scripts/generate_pretrained_report.py`**

```python
"""Generate a markdown report combining model rankings and pretrained availability.

Usage:
    python scripts/generate_pretrained_report.py

Reads:
    artifacts/promising_models.json         (from research_analyzer.py)
    artifacts/pretrained_search_results.json (from Task 1 web search)

Writes:
    artifacts/pretrained_models_report.md
"""
from __future__ import annotations
import json
from pathlib import Path

ARTIFACTS = Path(__file__).parent.parent / "artifacts"


def _availability_icon(search: dict) -> str:
    avail = search.get("pretrained_available")
    if avail is True:
        return "✅"
    if avail is False:
        return "❌"
    return "❓"


def main() -> None:
    models_path = ARTIFACTS / "promising_models.json"
    search_path = ARTIFACTS / "pretrained_search_results.json"

    if not models_path.exists():
        raise FileNotFoundError(f"{models_path} not found — run research_analyzer.py first")
    if not search_path.exists():
        raise FileNotFoundError(f"{search_path} not found — run Task 1 web search first")

    models_data = json.loads(models_path.read_text(encoding="utf-8"))
    search_data = json.loads(search_path.read_text(encoding="utf-8"))
    top_models = models_data["top_models"]

    lines: list[str] = [
        "# MRI Denoising & Super-Resolution: Promising Models with Pretrained Availability",
        "",
        f"Analyzed **{models_data['total_found']}** relevant papers from `research-organized/`.  ",
        f"Showing top {len(top_models)} models ranked by promise score + approach + code availability.",
        "",
        "## Summary Table",
        "",
        "| Rank | Model | Task | Promise | GitHub | HuggingFace | Pretrained |",
        "|------|-------|------|---------|--------|-------------|------------|",
    ]

    for i, m in enumerate(top_models, 1):
        name = m.get("model_name", "Unknown")
        search = search_data.get(name, {})
        gh_cell = f"[link]({search['github']})" if search.get("github") else "—"
        hf_cell = f"[link]({search['huggingface']})" if search.get("huggingface") else "—"
        icon = _availability_icon(search) if search else "❓"
        lines.append(
            f"| {i} | **{name}** | {m.get('task', '?')} | {m.get('promise', '?')}/5 "
            f"| {gh_cell} | {hf_cell} | {icon} |"
        )

    lines += ["", "---", "", "## Model Details", ""]

    for m in top_models:
        name = m.get("model_name", "Unknown")
        search = search_data.get(name, {})
        lines += [
            f"### {name}",
            "",
            f"**Task:** {m.get('task', '?')} | "
            f"**Approach:** {m.get('approach', '?')} | "
            f"**Promise:** {m.get('promise', '?')}/5 | "
            f"**Score:** {m.get('score', 0):.1f}",
            "",
            f"**Key method:** {m.get('key_method', '?')}",
            "",
            f"**Summary:** {m.get('summary', '?')}",
            "",
        ]
        if search:
            if search.get("github"):
                lines.append(f"**GitHub:** <{search['github']}>")
            if search.get("huggingface"):
                lines.append(f"**HuggingFace:** <{search['huggingface']}>")
            if search.get("papers_with_code"):
                lines.append(f"**Papers with Code:** <{search['papers_with_code']}>")
            icon = _availability_icon(search)
            notes = search.get("pretrained_notes") or "No weights found"
            lines.append(f"**Pretrained:** {icon} {notes}")
            if search.get("license"):
                lines.append(f"**License:** {search['license']}")
        else:
            lines.append("**Pretrained:** ❓ Not searched")
        lines += ["", "---", ""]

    report = "\n".join(lines)
    out = ARTIFACTS / "pretrained_models_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Report saved → {out}")
    print(f"  {sum(1 for m in top_models if search_data.get(m.get('model_name',''), {}).get('pretrained_available'))} / {len(top_models)} models have pretrained weights")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_generate_report.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Generate the actual report**

```powershell
python scripts/generate_pretrained_report.py
```

Expected output:
```
Report saved → artifacts/pretrained_models_report.md
  N / 15 models have pretrained weights
```

Open `artifacts/pretrained_models_report.md` and verify:
- Table has a row per model with icons (✅/❌/❓)
- Detail sections include GitHub/HuggingFace links where found

- [ ] **Step 6: Commit**

```powershell
git add scripts/generate_pretrained_report.py tests/test_generate_report.py artifacts/pretrained_models_report.md
git commit -m "feat: add report generator for pretrained model availability"
```

---

## Verification — End-to-End

```powershell
# 1. Analyze research output
python scripts/research_analyzer.py
# Expected: "Found N relevant models" with top-10 list

# 2. Generate report (uses Task 1's pretrained_search_results.json)
python scripts/generate_pretrained_report.py
# Expected: report saved with pretrained counts

# 3. Full test suite
python -m pytest tests/test_research_analyzer.py tests/test_generate_report.py -v
# Expected: all tests pass

# 4. View report
Get-Content artifacts/pretrained_models_report.md | Select-Object -First 50
```

---

## Notes / Risks

- **`research-organized/` is at workspace root** (`C:\projetos\research-organized\`), not inside `MRI_Denoise/`. The env var `RESEARCH_DIR` allows overriding; the default hardcodes the correct absolute path.
- **Task classification is inconsistent**: some files have `**Task:** Denoising` while others say `**Task:** Other` even if in a `Denoising/` folder. `is_relevant()` checks both the Task field and the folder path to catch both cases.
- **Duplicate papers**: the pipeline sometimes writes the same paper to multiple folders (e.g., a denoising paper tagged under both `Denoising` and `Segmentation`). `seen_titles` deduplicates by lowercased title prefix.
- **Short `.md` files**: some conference abstract files have very short content (2–3 lines). The parser is robust to missing fields — `parse_md_file` returns `None` only if title is absent.
- **Web search rate limits**: if Task 1 hits rate limits, search in batches of 5 with brief pauses. The JSON can be built incrementally.
- **Promise score bias**: the pipeline always assigns a score; scores of 3+/5 are the useful floor — models scoring below 3 are unlikely worth pursuing.
