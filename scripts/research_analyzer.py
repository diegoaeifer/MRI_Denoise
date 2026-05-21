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
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
