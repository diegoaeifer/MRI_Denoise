import torch
import torch.nn as nn

class AntigravityModel(nn.Module):
    """Dummy implementation of the theoretical Antigravity model."""
    def __init__(self, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class AntigravityAdapter(nn.Module):
    """
    Zero-mod adapter that fits the Antigravity model into the 2-channel MRI pipeline.
    """
    def __init__(self):
        super().__init__()
        self.adapter = nn.Conv2d(2, 1, kernel_size=1) # 2-channel to 1-channel fusion
        self.core = AntigravityModel(channels=1)

    def forward(self, x):
        # x is (B, 2, H, W) -> [Noisy Image, Sigma Map]
        x_fused = self.adapter(x)
        return self.core(x_fused)
