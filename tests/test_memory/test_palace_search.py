import pytest
from unittest.mock import patch
from src.mri_denoise.memory.palace_search import search_memory, MemoryResult


def test_memory_result_dataclass():
    r = MemoryResult(content="test content", score=0.9, room="models", wing="mri-denoise")
    assert r.content == "test content"
    assert r.score == 0.9


def test_search_memory_returns_list():
    fake_results = [
        {"document": "NAFNet was trained with lr=5e-4", "distance": 0.1,
         "metadata": {"room": "models", "wing": "mri-denoise"}},
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
        {"document": f"result {i}", "distance": float(i) * 0.1,
         "metadata": {"room": "r", "wing": "w"}}
        for i in range(10)
    ]
    with patch("src.mri_denoise.memory.palace_search._run_cli_search", return_value=fake_results[:3]):
        results = search_memory("query", top_k=3)
    assert len(results) <= 3
