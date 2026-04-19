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
            # We must be careful about dimension mapping.
            # The user's original update_keyvals fails on 3D tensors because they wrote it for 2D.
            # "for _ in range(new_value.shape[-1]): new_value[..., _] = old_value / new_value.shape[-1]"
            # In 3D (which old_value already is!), the shapes might actually just match.
            # Wait! Is old_value 3D or 2D? Let's print its shape.
            new_ckpt[old_key] = old_value.detach()
        else:
            if 'head' in old_key:
                c_in = new_value.shape[1]
                c_in_old = old_value.shape[1]
                if c_in == c_in_old:
                    new_value = old_value.detach()
                elif c_in < c_in_old:
                    new_value = torch.zeros_like(new_value.detach())
                    # In 3D, old_value is [C_out, C_in, D, H, W] -> 5D!
                    # If C_in is different, we can just copy
                    new_value[:, -1:, ...] = old_value[:, -1:, ...] # Noise channel
                    new_value[:, :c_in - 1, ...] = old_value[:, :c_in - 1, ...] # Image channels
            elif 'tail' in old_key:
                c_out = new_value.shape[0]
                c_out_old = old_value.shape[0]
                if c_out == c_out_old:
                    new_value = old_value.detach()
                elif c_out < c_out_old:
                    new_value = torch.zeros_like(new_value.detach())
                    new_value[:1, ...] = old_value[:1, ...]
                    new_value[1:, ...] = old_value[1:c_out, ...]
            new_ckpt[old_key] = new_value
    return new_ckpt

old_ckpt = torch.load('drunet_3d_complex_denoise.pth', map_location='cpu', weights_only=True)
print(f"Old head shape: {old_ckpt['m_head.weight'].shape}")
drunet = deepinv.models.DRUNet(in_channels=1, out_channels=1, dim=3, pretrained=None)
new_ckpt = deepcopy(drunet.state_dict())
print(f"New head shape: {new_ckpt['m_head.weight'].shape}")

new_ckpt = update_keyvals(old_ckpt, new_ckpt)
print("Success")
