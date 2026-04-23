## VisNet Integration

### Description
This pull request integrates the `VisNet` architecture (specifically the `DPN` model) from [yuehailin/VisNet](https://github.com/yuehailin/VisNet/tree/main/X-ray%20and%20MRI) into our local model factory for 2D MRI denoising.

### Changes Made:
- **`src/models/visnet.py`**: Added the adapted `DPN` model with all its auxiliary blocks (`BioInspiredLittleAddInhibitionBlock`, `Mdense_conv`, `DeformConv2d`, `Edge_Net`, `Spatial_attn_layer`, `Ventral_conv`, `Dorsal_conv`, `Fusion_conv`).
- Parameters such as `in_channels` and `out_channels` have been parameterized in the core network components (like `mdense1` and `fusion`) so they properly match the project's config file (in_channels=2, out_channels=1).
- Added `visnet` entry inside `configs/config_model.yaml` for testing/training configuration.
- Exposed `VisNet` in `src/models/factory.py`.
- **Bugfixes:**
  - `src/losses/auxiliary.py`: Cleaned up severe merge conflict markers (`<<<<<<< HEAD`) from a previous unresolved edit.
  - `src/data/loader.py`: Removed repetitive, broken boilerplate empty lines creating bad class initializations in `DICOMLoader`.
  - `src/models/ffdnet.py`: Created a stub file to prevent import errors in `factory.py` (which were failing all automated tests).
