import torch
import torch.nn as nn


class BaseMRIModel(nn.Module):
    """
    Standard interface for MRI Denoising models.
    Expects input x to be (B, 2, H, W) where:
    - x[:, 0] is the Noisy Image
    - x[:, 1] is the Sigma Map
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward")

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())
