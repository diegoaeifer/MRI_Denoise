import subprocess, sys
from pathlib import Path
import pytest

@pytest.mark.skipif(
    not Path("configs/config_lora_mri_finetune.yaml").exists(),
    reason="LoRA config not found"
)
def test_train_adapter_dry_run_lora():
    r = subprocess.run(
        [sys.executable, "scripts/train_adapter.py",
         "--config", "configs/config_lora_mri_finetune.yaml",
         "--dry_run"],
        capture_output=True, text=True, cwd="C:/projetos/MRI_Denoise"
    )
    assert r.returncode == 0, r.stderr
    assert "Trainable parameters" in r.stdout
    assert "lora" in r.stdout.lower()

@pytest.mark.skipif(
    not Path("configs/config_foura_mri_finetune.yaml").exists(),
    reason="FouRA config not found"
)
def test_train_adapter_dry_run_foura():
    r = subprocess.run(
        [sys.executable, "scripts/train_adapter.py",
         "--config", "configs/config_foura_mri_finetune.yaml",
         "--dry_run"],
        capture_output=True, text=True, cwd="C:/projetos/MRI_Denoise"
    )
    assert r.returncode == 0, r.stderr
    assert "Trainable parameters" in r.stdout
    assert "foura" in r.stdout.lower()
