import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def fake_artifacts():
    """Provide minimal JSON fixtures so the script runs without real artifacts."""
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
    subprocess.run([sys.executable, "scripts/generate_pretrained_report.py"],
                   capture_output=True, text=True, cwd=str(REPO))
    text = (REPO / "artifacts" / "pretrained_models_report.md").read_text(encoding="utf-8")
    assert "| Rank |" in text
    assert "| 1 |" in text


def test_report_pretrained_checkmark():
    subprocess.run([sys.executable, "scripts/generate_pretrained_report.py"],
                   capture_output=True, text=True, cwd=str(REPO))
    text = (REPO / "artifacts" / "pretrained_models_report.md").read_text(encoding="utf-8")
    assert "✅" in text   # GAMBAS has pretrained_available=True
    assert "❌" in text   # Noise2Average has pretrained_available=False


def test_report_contains_detail_sections():
    subprocess.run([sys.executable, "scripts/generate_pretrained_report.py"],
                   capture_output=True, text=True, cwd=str(REPO))
    text = (REPO / "artifacts" / "pretrained_models_report.md").read_text(encoding="utf-8")
    assert "### GAMBAS" in text
    assert "### Noise2Average" in text
