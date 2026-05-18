"""Wrapper for Restore-RWKV (Yaziwel/Restore-RWKV) model.

Adapts (B, 2, H, W) input (noisy image + sigma map) → (B, 1, H, W) output.

The underlying Restore_RWKV expects:
  - inp_img: (B, C, H, W)  where C == inp_channels (default 1)
  - label:   optional target tensor (omit during inference)

Important: Restore_RWKV compiles a CUDA extension (bi_wkv) at import time.
This requires a working CUDA toolkit.  If compilation fails the import is
skipped and an informative ImportError is raised on forward().

Forward convention used here:
  ch0 → noisy image  (passed to model)
  ch1 → sigma map    (not used by Restore-RWKV; ignored)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

_RWKV_REPO = Path(__file__).parent.parent.parent / "FMImaging_MRI_Denoise" / "Restore-RWKV"


def _import_restore_rwkv():
    """Lazily add repo to sys.path and import Restore_RWKV.  Returns the class."""
    repo_str = str(_RWKV_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        from model.Restore_RWKV import Restore_RWKV  # noqa: PLC0415
        return Restore_RWKV
    except ImportError as exc:
        raise ImportError(
            "Restore_RWKV could not be imported (CUDA extension compilation may have failed). "
            f"Make sure the repo is cloned at {_RWKV_REPO} and a CUDA toolkit is available. "
            "Run: git clone https://github.com/Yaziwel/Restore-RWKV "
            f"{_RWKV_REPO}"
        ) from exc


class RestoreRWKVWrapper(nn.Module):
    """Wraps Restore_RWKV to accept (B, 2, H, W) factory-standard input.

    Parameters
    ----------
    in_channels:
        Must be 2 (image + sigma map).  Kept for API symmetry.
    dim:
        Feature width inside Restore_RWKV (default 48, as in paper).
    num_blocks:
        Encoder block counts per scale level.
    num_refinement_blocks:
        Number of refinement blocks after the decoder.
    """

    def __init__(
        self,
        in_channels: int = 2,
        dim: int = 48,
        num_blocks: list = None,
        num_refinement_blocks: int = 4,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        Restore_RWKV = _import_restore_rwkv()
        # Restore_RWKV takes 1-channel input; sigma channel is discarded
        self.net = Restore_RWKV(
            inp_channels=1,
            out_channels=1,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, H, W)
            ch0 = noisy image, ch1 = sigma map (ignored by this model)

        Returns
        -------
        (B, 1, H, W) denoised image
        """
        img = x[:, :1, :, :]   # (B, 1, H, W)
        # Restore_RWKV returns tensor directly (when label=None)
        return self.net(img)
