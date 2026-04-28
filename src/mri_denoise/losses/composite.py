import torch
import torch.nn as nn
from monai.losses import SSIMLoss, PerceptualLoss
from .epi import EPILoss

class CompositeLoss(nn.Module):
    def __init__(self, spatial_dims: int, weights: dict, data_range: float = 1.0):
        super().__init__()
        self.weights = weights
        self.components = nn.ModuleDict({
            "l1": nn.L1Loss(),
            "ssim": SSIMLoss(spatial_dims=spatial_dims, data_range=data_range),
            "epi": EPILoss()
        })

        if self.weights.get("perceptual", 0.0) > 0:
            self.components["perceptual"] = PerceptualLoss(
                spatial_dims=spatial_dims, network_type="vgg"
            )

    def forward(self, pred, target):
        total = 0.0
        details = {}

        # Determine effective spatial dims based on input shape
        # (pred might be 5D, PerceptualLoss might need adjusting if 3D)

        for name, w in self.weights.items():
            if w == 0: continue
            if name not in self.components: continue

            # Perceptual loss expects 3 channels
            if name == "perceptual" and pred.shape[1] == 1:
                p_vgg = pred.repeat(1, 3, 1, 1) if pred.ndim == 4 else pred.repeat(1, 3, 1, 1, 1)
                t_vgg = target.repeat(1, 3, 1, 1) if target.ndim == 4 else target.repeat(1, 3, 1, 1, 1)
                val = self.components[name](p_vgg, t_vgg)
            else:
                val = self.components[name](pred, target)

            total = total + w * val
            details[name] = val.detach()

        return total, details
