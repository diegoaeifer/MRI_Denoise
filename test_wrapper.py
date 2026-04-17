import torch
import deepinv
import torch.nn as nn
from copy import deepcopy
from update_keyvals import update_keyvals
import os
import urllib.request

class DeepinvDRUNet3DPretrained(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        # deepinv expects pure image channels, so subtract 1 for the noise map
        self.image_channels = in_channels - 1 if in_channels > 1 else in_channels
        self.out_channels = out_channels

        self.drunet = deepinv.models.DRUNet(
            in_channels=self.image_channels,
            out_channels=self.out_channels,
            dim=3,
            pretrained=None
        )

        # Download and load weights
        weight_path = "weights/drunet/drunet_3d_complex_denoise.pth"
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        if not os.path.exists(weight_path):
            url = "https://huggingface.co/deepinv/drunet_3d_denoise_complex/resolve/main/drunet_3d_complex_denoise.pth"
            urllib.request.urlretrieve(url, weight_path)

        ckpt = torch.load(weight_path, map_location='cpu', weights_only=True)
        new_ckpt = deepcopy(self.drunet.state_dict())
        new_ckpt = update_keyvals(ckpt, new_ckpt)
        self.drunet.load_state_dict(new_ckpt)

    def forward(self, x):
        # x is (B, in_channels, D, H, W). Last channel is sigma map.
        img = x[:, :-1, ...]
        sigma_map = x[:, -1:, ...]

        # Because deepinv.models.DRUNet has a bug with 3D sigma expansion when passing a scalar/tensor,
        # and we already have a sigma_map of full spatial dimensions, we can just concatenate it ourselves
        # and call forward_unet directly.
        x_with_noise = torch.cat((img, sigma_map), dim=1)

        return self.drunet.forward_unet(x_with_noise)

wrapper = DeepinvDRUNet3DPretrained(2, 1)
x = torch.randn(1, 2, 16, 16, 16)
out = wrapper(x)
print(out.shape)
