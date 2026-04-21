# Antigravity Model Integration & Testing Plan

## 1. Code Boundary Analysis
**Strict Zero-Modification Policy:** We will integrate the Antigravity model using the **External Adapter Pattern**.

### Do Not Touch List (Strictly Off-Limits)
- `src/models/factory.py` (No registering the model here)
- `src/train.py` (No modifying the training loop)
- `configs/*.yaml` (No adding new keys here)
- `tests/*` (No polluting the main test suite)
- `requirements.txt` (Assume dependencies are met or isolated)

### Touch / Interaction Points
- **Directory:** A completely isolated `antigravity_integration/` directory at the project root.
- **Data Loaders:** We will cleanly import `src.data.dataset.MRI_DICOM_Dataset` and `src.data.loader.DICOMLoader` into our isolated scripts.
- **Trainer:** We will instantiate `src.trainer.Trainer` in our isolated scripts, passing it our dynamically injected Antigravity adapter.
- **Loss:** `src.losses.composite.CompositeLoss` will be imported for standalone metric evaluation.

## 2. Hardware-Aware Architectural Decisions
**Target Specs:** 16GB VRAM, 32GB RAM.
- **Batch Sizing:** Antigravity testing will use a strict `batch_size=4` to avoid exceeding the 16GB VRAM threshold during backward passes, given the complex 16-bit MRI floating point requirements.
- **Gradient Checkpointing:** Recommended inside the Antigravity model if intermediate feature maps exceed 8GB.
- **Mixed Precision:** We will enable `torch.cuda.amp.autocast` in our test adapter to simulate production efficiency and validate numerical stability (FP16/FP32).

## 3. Step-by-step Setup
1. Define the Antigravity dummy model and adapter inside `antigravity_integration/adapter.py`.
2. Map the 2-channel `(image, sigma_map)` input to the Antigravity's expected tensor format using a transparent `nn.Module` wrapper.
3. Build the isolated PyTest suite in `antigravity_integration/tests_antigravity.py`.
4. Execute via the shell script `antigravity_integration/run_tests.sh`.

## 4. Commands to Run GPU Tests
To run the fully isolated Antigravity test suite:
```bash
cd antigravity_integration
bash run_tests.sh
```
