import torch
import deepinv
from copy import deepcopy

def update_keyvals(old_ckpt, new_ckpt):
    """
    Converting old DRUNet keys to new UNExt style keys.
    KEYS do not change but weight need to be 0 padded.
    """
    for old_key, old_value in old_ckpt.items():
        new_value = new_ckpt[old_key]
        if not 'head' in old_key and not 'tail' in old_key:
            for _ in range(new_value.shape[-1]):
                new_value[..., _] = old_value / new_value.shape[-1]
            new_ckpt[old_key] = new_value
        else:
            if 'head' in old_key:
                c_in = new_value.shape[1]
                c_in_old = old_value.shape[1]
                if c_in == c_in_old:
                    new_value = old_value.detach()
                elif c_in < c_in_old:
                    new_value = torch.zeros_like(new_value.detach())
                    for _ in range(new_value.shape[-1]):
                        new_value[:, -1:, ..., _] = old_value[:, -1:, ...] / new_value.shape[-1]
                    for _ in range(new_value.shape[-1]):
                        new_value[:, :c_in - 1, ..., _] = old_value[:, :c_in - 1, ...] / new_value.shape[-1]
            elif 'tail' in old_key:
                c_in = new_value.shape[0]
                c_in_old = old_value.shape[0]
                new_value = torch.zeros_like(new_value.detach())
                if c_in == c_in_old:
                    new_value = old_value.detach()
                elif c_in < c_in_old:
                    new_value = torch.zeros_like(new_value.detach())
                    for _ in range(new_value.shape[-1]):
                        new_value[:1, ..., _] = old_value[:1, ...] / new_value.shape[-1]
                    for _ in range(new_value.shape[-1]):
                        new_value[1:, ..., _] = old_value[1:c_in, ...] / new_value.shape[-1]
            new_ckpt[old_key] = new_value
    return new_ckpt

old_ckpt = torch.load('drunet_3d_complex_denoise.pth', map_location='cpu', weights_only=True)
drunet = deepinv.models.DRUNet(in_channels=1, out_channels=1, dim=3, pretrained=None)
new_ckpt = deepcopy(drunet.state_dict())

new_ckpt = update_keyvals(old_ckpt, new_ckpt)
print("Success")
