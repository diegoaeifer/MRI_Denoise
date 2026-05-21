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
