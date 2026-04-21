import pytest
import torch
import os
import sys
import time

# Pre-add project root to sys.path for isolated imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from antigravity_integration.adapter import AntigravityAdapter

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def mock_input(device):
    # (Batch, Channels, Height, Width) -> 16GB VRAM safe batch=4, size=256
    return torch.randn(4, 2, 256, 256, device=device)

@pytest.fixture
def mock_edge_input(device):
    # Edge case: Max hardware safe size for single inference
    return torch.randn(1, 2, 1024, 1024, device=device)

def test_antigravity_forward_pass_shape(device, mock_input):
    """Integration Test: Verifies adapter accurately transforms dimensions."""
    model = AntigravityAdapter().to(device)
    model.eval()

    with torch.no_grad():
        output = model(mock_input)

    assert output.shape == (4, 1, 256, 256), f"Expected shape (4, 1, 256, 256), got {output.shape}"

def test_antigravity_edge_case_large_image(device, mock_edge_input):
    """Integration Test: Verifies adapter can handle massive inputs safely."""
    model = AntigravityAdapter().to(device)
    model.eval()

    with torch.no_grad():
        output = model(mock_edge_input)

    assert output.shape == (1, 1, 1024, 1024)

def test_antigravity_edge_case_zeros(device):
    """Unit Test: Model handles completely flat (black) images without NaN."""
    model = AntigravityAdapter().to(device)
    flat_input = torch.zeros(2, 2, 64, 64, device=device)
    output = model(flat_input)
    assert not torch.isnan(output).any(), "NaN detected on zero input"

def test_antigravity_gpu_memory_leak(device, mock_input):
    """Memory Profiling: Ensures model forward/backward doesn't leak VRAM."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for memory leak test.")

    model = AntigravityAdapter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats(device)
    initial_mem = torch.cuda.memory_allocated(device)

    for _ in range(5):
        optimizer.zero_grad()
        out = model(mock_input)
        loss = out.mean()
        loss.backward()
        optimizer.step()

    final_mem = torch.cuda.memory_allocated(device)

    # Allow for some optimizer state allocation, but prevent unbound leaks
    mem_diff_mb = (final_mem - initial_mem) / (1024 ** 2)
    assert mem_diff_mb < 500.0, f"Memory leaked {mem_diff_mb} MB during short loop"

def test_antigravity_gradient_flow(device, mock_input):
    """Unit Test: Ensures gradients propagate correctly through adapter to core."""
    model = AntigravityAdapter().to(device)

    out = model(mock_input)
    loss = out.sum()
    loss.backward()

    assert model.adapter.weight.grad is not None, "Gradients did not flow to adapter"
    assert model.core.net[0].weight.grad is not None, "Gradients did not flow to core model"

def test_antigravity_performance_profile(device, mock_input):
    """Performance Profiling: Ensures model inference meets speed targets."""
    model = AntigravityAdapter().to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        _ = model(mock_input)

    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(mock_input)

    end_time = time.time()
    avg_latency = (end_time - start_time) / 10
    assert avg_latency < 1.0, f"Inference too slow: {avg_latency} seconds/batch"

def test_antigravity_autocast_stability(device, mock_input):
    """Integration Test: Validates Mixed Precision stability (FP16/FP32)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for autocast test.")

    model = AntigravityAdapter().to(device)
    model.eval()

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            output = model(mock_input)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
