# ============================================================
# install_cuda_torch.ps1
# Run this ONCE after installing CUDA to upgrade to the
# CUDA-enabled PyTorch build (CUDA 12.1 compatible).
# Change the index URL if your CUDA version differs.
# ============================================================

Write-Host "Uninstalling CPU-only torch..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

Write-Host "Installing CUDA-enabled torch (CUDA 12.1)..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Verifying CUDA availability..." -ForegroundColor Cyan
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
