import torch
import torch.nn as nn
import deepinv

class RAMWrapper(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(RAMWrapper, self).__init__()
        # Initialize DeepInv RAM
        self.model = deepinv.models.RAM()

    def forward(self, x):
        # x is (B, 2, H, W) where channel 0 is noisy image, channel 1 is sigma
        # RAM expects y (noisy image) and sigma as arguments
        y = x[:, 0:1, :, :]

        # Sigma map is provided in channel 1.
        # Deepinv RAM accepts a scalar or a tensor for sigma.
        # We calculate the mean sigma for each item in the batch.
        sigma_map = x[:, 1:2, :, :]

        # Taking mean over spatial dims (H, W) and channel (1) to get (B,)
        sigma = sigma_map.mean(dim=(1, 2, 3))

        # RAM also accepts 'gain' for Poisson noise, but since this is MRI we just pass sigma.
        out = self.model(y, sigma=sigma, gain=torch.zeros_like(sigma))
        return out
