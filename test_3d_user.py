import torch
import deepinv

drunet = deepinv.models.DRUNet(in_channels=2, out_channels=2, dim=3, pretrained=None)
print("Keys:", list(drunet.state_dict().keys())[:5])
