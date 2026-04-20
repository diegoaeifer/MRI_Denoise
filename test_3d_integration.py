import torch
import deepinv

def get_keys():
    drunet = deepinv.models.DRUNet(in_channels=2, out_channels=2, dim=3, pretrained=None)
    x = torch.randn(1, 2, 16, 16, 16)
    sigma = torch.tensor([0.1])
    # The default DRUNet implementation in deepinv assumes sigma expansion to 4D!
    # Let's see the error again.
    try:
        out = drunet(x, sigma)
    except Exception as e:
        print(e)

get_keys()
