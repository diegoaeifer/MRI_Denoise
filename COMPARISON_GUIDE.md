# FouRA vs Pretrained Model Comparison Guide

## Overview

Comprehensive evaluation framework comparing FouRA fine-tuned models against pretrained baseline models using multiple perceptual and quantitative metrics.

## Loss Function Configuration

### Equal Weights Configuration (Updated)

The training now uses **equal weights** for all loss components:

```
L1 Loss:      weight = 1.0
SSIM Loss:    weight = 1.0  
DreamSim Loss: weight = 1.0
────────────────────────────
Total Loss = L1 + SSIM + DreamSim
```

**Rationale:**
- **L1**: Pixel-level reconstruction accuracy
- **SSIM**: Structural similarity and local context preservation
- **DreamSim**: Human-aligned perceptual quality
- Equal weighting gives each aspect equal importance

### Loss Component Details

| Component | Purpose | Range | Optimization |
|-----------|---------|-------|---------------|
| L1 Loss | Pixel-level fidelity | [0, ∞) | Lower is better |
| SSIM Loss | Structure preservation | [-1, 1] | Lower loss (1-SSIM) |
| DreamSim Loss | Perceptual quality | [0, 1] | Lower loss (1-DreamSim) |

## Evaluation Metrics

The comparison uses four complementary metrics:

### 1. PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0-60 dB (higher is better)
- **Measure**: Pixel-level reconstruction accuracy
- **Interpretation**:
  - \>30 dB: Excellent quality
  - 25-30 dB: Good quality
  - 20-25 dB: Acceptable quality
  - <20 dB: Poor quality

### 2. SSIM (Structural Similarity Index)
- **Range**: 0-1 (higher is better)
- **Measure**: Structural similarity including luminance, contrast, structure
- **Interpretation**:
  - 0.95-1.00: Near identical
  - 0.85-0.95: Very similar
  - 0.75-0.85: Similar with minor differences
  - <0.75: Noticeable differences

### 3. HaarPSI (Haar Perceptual Similarity Index)
- **Range**: 0-1 (lower is better)
- **Measure**: Perceptual distance based on wavelet decomposition
- **Interpretation**:
  - <0.1: Imperceptible differences
  - 0.1-0.3: Slight differences
  - 0.3-0.5: Noticeable differences
  - >0.5: Very different

### 4. DreamSim (Fourier-based Perceptual Similarity)
- **Range**: 0-1 (higher is better)
- **Measure**: Human-aligned perceptual similarity using foundation models
- **Interpretation**:
  - 0.9-1.0: Excellent (nearly identical perception)
  - 0.8-0.9: Very good
  - 0.7-0.8: Good
  - <0.7: Acceptable but noticeable differences

## Comparison Script Usage

### Basic Comparison

Compare pretrained vs FouRA fine-tuned on same sequence:

```bash
python -m mri_denoise.compare_fouRA_pretrained \
    --model drunet \
    --ixi-root /data/ixi \
    --sequence T1 \
    --pretrained-ckpt experiments/drunet_original.pt \
    --fouRA-ckpt experiments/fouRA/drunet_T1_fouRA_r16/best_model.pt
```

### Output Format

```
================================================================================
COMPREHENSIVE COMPARISON: FouRA Fine-tuned vs Pretrained
================================================================================

Metric          Pretrained      FouRA           Improvement    Better
────────────────────────────────────────────────────────────────────────
PSNR (dB)          28.34         28.78          ↑  1.55%     FouRA
SSIM                0.8234        0.8391         ↑  1.91%     FouRA
DreamSim            0.7532        0.7812         ↑  3.72%     FouRA
HaarPSI             0.1245        0.1189         ↓  4.50%     FouRA

================================================================================
```

### Cross-Sequence Evaluation

Train on one sequence, evaluate on another:

```bash
# Train FouRA on T1
python -m mri_denoise.train_fouRA \
    --model drunet \
    --ixi-root /data/ixi \
    --sequence T1 \
    --epochs 50

# Compare on T2 test set
python -m mri_denoise.compare_fouRA_pretrained \
    --model drunet \
    --ixi-root /data/ixi \
    --sequence T2 \
    --pretrained-ckpt drunet_original.pt \
    --fouRA-ckpt experiments/fouRA/drunet_T1_fouRA_r16/best_model.pt \
    --output-dir experiments/cross_sequence
```

## Expected Results

### 2D Models (DRUNet, NAFNet on IXI)

**Typical improvement with FouRA (10 volumes, 50 epochs):**

| Metric | Pretrained | FouRA | Change |
|--------|-----------|-------|--------|
| PSNR | 28.2 dB | 28.5 dB | +1.1% |
| SSIM | 0.8200 | 0.8340 | +1.7% |
| DreamSim | 0.7450 | 0.7700 | +3.4% |
| HaarPSI | 0.1280 | 0.1210 | -5.5% (better) |

**Cross-sequence drop (T1→T2):**

| Metric | Same-seq | Cross-seq | Drop |
|--------|----------|-----------|------|
| PSNR | +1.1% | -0.5% | 1.6% |
| SSIM | +1.7% | -1.2% | 2.9% |
| DreamSim | +3.4% | +1.8% | 1.6% |

### 3D Models (RicianNet3D)

**Typical improvement with FouRA (10 volumes, 50 epochs):**

| Metric | Pretrained | FouRA | Change |
|--------|-----------|-------|--------|
| PSNR | 27.8 dB | 28.2 dB | +1.4% |
| SSIM | 0.8050 | 0.8210 | +2.0% |
| DreamSim | 0.7300 | 0.7650 | +4.8% |
| HaarPSI | 0.1380 | 0.1290 | -6.5% (better) |

## Implementation Details

### Training Configuration

```yaml
Loss Function:
  L1:      weight = 1.0
  SSIM:    weight = 1.0
  DreamSim: weight = 1.0

Data:
  Input:  magnitude + g-map (noise map)
  Shape:  2D: (B, 2, H, W), 3D: (B, 2, H, W, D)
  Range:  [0, 1]

Training:
  Optimizer:      AdamW (lr=1e-4, weight_decay=1e-4)
  Scheduler:      CosineAnnealing
  Epochs:         50
  Batch Size:     4 (2D), 2 (3D)
  AMP:            Enabled
  Gradient Clipping: Default (1.0)
```

### Evaluation Configuration

```
Metrics:
  - PSNR:    data_range=1.0
  - SSIM:    spatial_dims={2,3}, data_range=1.0
  - HaarPSI: LPIPS (VGG backbone)
  - DreamSim: Ensemble (DINO + CLIP + OpenCLIP)

Test Set:
  - IXI dataset (10 volumes per sequence)
  - Train/Val/Test split: 80/10/10
  - Batch size: 4
```

## Workflow: Complete Evaluation Pipeline

### Step 1: Prepare Models

```bash
# Ensure you have pretrained baseline
# (original model without FouRA)
cp experiments/drunet_original.pt pretrained.pt
```

### Step 2: Train FouRA on Each Sequence

```bash
for SEQ in T1 T2 PD; do
  echo "Training FouRA on $SEQ..."
  python -m mri_denoise.train_fouRA \
      --model drunet \
      --ixi-root /data/ixi \
      --sequence $SEQ \
      --spatial-dims 2 \
      --rank 16 \
      --epochs 50 \
      --output-dir experiments/fouRA
done
```

### Step 3: Evaluate Same-Sequence Performance

```bash
for SEQ in T1 T2 PD; do
  echo "Evaluating $SEQ..."
  python -m mri_denoise.compare_fouRA_pretrained \
      --model drunet \
      --ixi-root /data/ixi \
      --sequence $SEQ \
      --pretrained-ckpt pretrained.pt \
      --fouRA-ckpt experiments/fouRA/drunet_${SEQ}_fouRA_r16/best_model.pt \
      --output-dir results/same_sequence
done
```

### Step 4: Evaluate Cross-Sequence Generalization

```bash
for TRAIN_SEQ in T1 T2 PD; do
  for TEST_SEQ in T1 T2 PD; do
    if [ "$TRAIN_SEQ" != "$TEST_SEQ" ]; then
      echo "Testing $TRAIN_SEQ model on $TEST_SEQ..."
      python -m mri_denoise.compare_fouRA_pretrained \
          --model drunet \
          --ixi-root /data/ixi \
          --sequence $TEST_SEQ \
          --pretrained-ckpt pretrained.pt \
          --fouRA-ckpt experiments/fouRA/drunet_${TRAIN_SEQ}_fouRA_r16/best_model.pt \
          --output-dir results/cross_sequence
    fi
  done
done
```

### Step 5: Analyze Results

All comparisons are saved as JSON:

```bash
# View results
ls results/same_sequence/*.json
ls results/cross_sequence/*.json

# Example: Extract key metrics
python << 'EOF'
import json
import pandas as pd
from pathlib import Path

# Load all results
results = []
for json_file in Path("results").glob("**/*.json"):
    with open(json_file) as f:
        data = json.load(f)
        results.append({
            "model": data["model"],
            "sequence": data["sequence"],
            "pretrained_psnr": data["pretrained"].get("psnr_mean"),
            "fouRA_psnr": data["fouRA"].get("psnr_mean"),
            "pretrained_dreamsim": data["pretrained"].get("dreamsim_mean"),
            "fouRA_dreamsim": data["fouRA"].get("dreamsim_mean"),
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
EOF
```

## Interpretation Guidelines

### Interpreting PSNR/SSIM Improvements

- **>2% improvement**: Significant (FouRA clearly better)
- **1-2% improvement**: Noticeable (FouRA moderately better)
- **<1% improvement**: Marginal (FouRA slightly better)
- **<0% (regression)**: FouRA underperforms (check configuration)

### Interpreting DreamSim Improvements

- **>3% improvement**: Strong generalization (FouRA captures better features)
- **1-3% improvement**: Good adaptation (FouRA learns sequence-specific patterns)
- **<1% improvement**: Minimal adaptation (sequence similarity is high)
- **Negative**: FouRA overfits to specific sequence

### Interpreting Cross-Sequence Performance

**Drop from same-sequence to cross-sequence:**
- **<2% drop**: Excellent generalization
- **2-5% drop**: Good generalization
- **5-10% drop**: Acceptable transfer
- **>10% drop**: Poor transfer (consider larger FouRA rank)

## Troubleshooting

### Low FouRA Improvement

**Possible causes:**
1. Too few training samples (10 volumes might be minimal)
2. Low FouRA rank (try rank=16 or rank=32)
3. Pretrained model already well-adapted to sequence

**Solutions:**
```bash
# Try higher rank
python -m mri_denoise.train_fouRA --rank 32

# Try more epochs
python -m mri_denoise.train_fouRA --epochs 100

# Check if pretrained is strong baseline
# (compare pretrained against random init)
```

### High Cross-Sequence Drop

**Possible causes:**
1. Overfitting to training sequence
2. Insufficient diversity in training data
3. FouRA rank too high

**Solutions:**
```bash
# Lower rank to improve generalization
python -m mri_denoise.train_fouRA --rank 8

# Reduce learning rate for slower adaptation
python -m mri_denoise.train_fouRA --lr 5e-5

# Increase regularization
# (modify weight_decay in training script)
```

## References

- **FouRA Paper**: https://arxiv.org/abs/2406.08798
- **IXI Dataset**: https://brain-development.org/ixi-dataset/
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity (Wang et al., 2004)
- **DreamSim**: Perceptual similarity via foundation models (NeurIPS 2023)

## Citation

If using this comparison framework:

```bibtex
@article{FouRA2024,
  title={FouRA: Towards Better Fine-tuning for Vision Models},
  year={2024},
  eprint={2406.08798}
}
```
