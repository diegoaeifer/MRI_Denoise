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
    print(f"Report saved -> {out}")
    print(f"  {sum(1 for m in top_models if search_data.get(m.get('model_name',''), {}).get('pretrained_available'))} / {len(top_models)} models have pretrained weights")


if __name__ == "__main__":
    main()
