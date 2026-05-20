import json, subprocess, sys
from pathlib import Path
import pytest

MANIFEST = Path("D:/Dataset MRI/manifest.csv")

@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not found at D:/Dataset MRI/manifest.csv")
def test_prepare_dataset_creates_split():
    r = subprocess.run(
        [sys.executable, "scripts/prepare_finetune_dataset.py",
         "--manifest", "D:/Dataset MRI/manifest.csv",
         "--out", "data/finetune_split.json",
         "--seed", "42"],
        capture_output=True, text=True, cwd="C:/projetos/MRI_Denoise"
    )
    assert r.returncode == 0, r.stderr
    with open("C:/projetos/MRI_Denoise/data/finetune_split.json") as f:
        split = json.load(f)
    assert "train" in split and "val" in split and "test" in split
    assert len(split["train"]) > 0
    # No subject appears in both train and test (composite key guards cross-dataset leakage)
    train_subjects = {(d["source_dataset"], d["subject_id"]) for d in split["train"]}
    test_subjects  = {(d["source_dataset"], d["subject_id"]) for d in split["test"]}
    assert train_subjects.isdisjoint(test_subjects), "Subject leakage!"
