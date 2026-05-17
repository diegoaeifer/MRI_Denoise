"""TDD tests for comprehensive_benchmark.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load():
    spec = importlib.util.spec_from_file_location(
        "comprehensive_benchmark",
        Path(__file__).parent.parent / "scripts" / "comprehensive_benchmark.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load()


class _FakeLoader:
    def __init__(self, dataset="TestDS", modality="T1"):
        self.dataset = dataset
        self.modality = modality

    def volumes(self):
        vol = np.random.rand(16, 16, 4).astype(np.float32)
        yield {
            "volume": vol,
            "dataset": self.dataset,
            "modality": self.modality,
            "subject_id": "sub001",
        }


class _IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :, :]


def test_run_benchmark_2d_produces_records():
    """One record per (model, sigma, modality) plus noisy_input baseline."""
    models = {"identity": _IdentityModel().eval()}
    records = cb.run_benchmark_2d(
        loaders=[_FakeLoader()],
        models=models,
        sigma_levels=[0.05],
        device=torch.device("cpu"),
        progress=set(),
    )
    model_names = {r["model"] for r in records}
    assert "identity" in model_names
    assert "noisy_input" in model_names


def test_run_benchmark_2d_record_has_all_metric_keys():
    models = {"identity": _IdentityModel().eval()}
    records = cb.run_benchmark_2d(
        loaders=[_FakeLoader()],
        models=models,
        sigma_levels=[0.05],
        device=torch.device("cpu"),
        progress=set(),
    )
    identity_records = [r for r in records if r["model"] == "identity"]
    assert identity_records, "no identity record produced"
    rec = identity_records[0]
    for key in ("psnr", "ssim", "haarpsi", "vgg_loss", "sharpness"):
        assert key in rec, f"missing key: {key}"


def test_run_benchmark_2d_psnr_is_finite_for_identity():
    """Identity model should yield finite PSNR (not inf, since Rician noise present)."""
    models = {"identity": _IdentityModel().eval()}
    records = cb.run_benchmark_2d(
        loaders=[_FakeLoader()],
        models=models,
        sigma_levels=[0.05],
        device=torch.device("cpu"),
        progress=set(),
    )
    identity_rec = next(r for r in records if r["model"] == "identity")
    assert np.isfinite(identity_rec["psnr"])
    assert identity_rec["psnr"] > 0


def test_run_benchmark_2d_skips_completed():
    """Records already in progress set are skipped."""
    models = {"identity": _IdentityModel().eval()}
    pre_done = {
        "noisy|noisy_input|TestDS|T1|0.05",
        "2d|identity|TestDS|T1|0.05",
    }
    records = cb.run_benchmark_2d(
        loaders=[_FakeLoader()],
        models=models,
        sigma_levels=[0.05],
        device=torch.device("cpu"),
        progress=pre_done,
    )
    assert records == []


def test_run_benchmark_2d_multiple_sigmas_produces_multiple_records():
    models = {"identity": _IdentityModel().eval()}
    records = cb.run_benchmark_2d(
        loaders=[_FakeLoader()],
        models=models,
        sigma_levels=[0.02, 0.10],
        device=torch.device("cpu"),
        progress=set(),
    )
    identity_records = [r for r in records if r["model"] == "identity"]
    assert len(identity_records) == 2  # one per sigma


def test_run_benchmark_2d_failed_model_is_skipped():
    """A model that raises an exception is skipped; other models still run."""
    class _BrokenModel(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("intentional failure")

    models = {
        "broken": _BrokenModel().eval(),
        "identity": _IdentityModel().eval(),
    }
    records = cb.run_benchmark_2d(
        loaders=[_FakeLoader()],
        models=models,
        sigma_levels=[0.05],
        device=torch.device("cpu"),
        progress=set(),
    )
    model_names = {r["model"] for r in records}
    assert "identity" in model_names
    assert "broken" not in model_names


def test_progress_key_and_load_save(tmp_path):
    """Checkpoint round-trips correctly."""
    done = {"2d|drunet|IXI|T1|0.05", "2d|drunet|IXI|T2|0.10"}
    pf = tmp_path / ".progress.json"
    cb.save_progress(pf, done)
    loaded = cb.load_progress(pf)
    assert loaded == done


def test_run_benchmark_2d_progress_file_grows(tmp_path):
    """After the run, progress set contains new keys."""
    models = {"identity": _IdentityModel().eval()}
    progress = set()
    cb.run_benchmark_2d(
        loaders=[_FakeLoader()],
        models=models,
        sigma_levels=[0.05],
        device=torch.device("cpu"),
        progress=progress,
    )
    pf = tmp_path / ".progress.json"
    cb.save_progress(pf, progress)
    loaded = cb.load_progress(pf)
    assert any("identity" in k for k in loaded)
    assert any("noisy_input" in k for k in loaded)


def test_generate_comprehensive_report_creates_html(tmp_path):
    """generate_comprehensive_report writes a non-empty HTML file."""
    spec = importlib.util.spec_from_file_location(
        "comprehensive_report",
        Path(__file__).parent.parent / "scripts" / "comprehensive_report.py",
    )
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)

    import pandas as pd

    df = pd.DataFrame([
        {"track": "2d", "model": "identity", "dataset": "IXI", "modality": "T1",
         "subject_id": "s001", "sigma": 0.05, "psnr": 30.0, "ssim": 0.9,
         "haarpsi": 0.85, "vgg_loss": 0.12, "sharpness": 0.004},
        {"track": "2d", "model": "noisy_input", "dataset": "IXI", "modality": "T1",
         "subject_id": "s001", "sigma": 0.05, "psnr": 20.0, "ssim": 0.7,
         "haarpsi": 0.60, "vgg_loss": 0.30, "sharpness": 0.003},
    ])
    csv_path = tmp_path / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    html_path = cr.generate_comprehensive_report(csv_path, tmp_path)
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "<html" in content
    assert "identity" in content
