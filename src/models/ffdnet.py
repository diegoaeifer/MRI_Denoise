import torch
import torch.nn as nn


class FFDNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_layers=15):
        super(FFDNet, self).__init__()
        # Stub implementation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, sigma):
        return self.conv(x)
