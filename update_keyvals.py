import torch

def update_keyvals(old_ckpt, new_ckpt):
    """
    Converting old DRUNet keys to new UNExt style keys.
    KEYS do not change but weight need to be 0 padded.
    """
    for old_key, old_value in old_ckpt.items():
        if old_key not in new_ckpt:
            continue
        new_value = new_ckpt[old_key]
        if not 'head' in old_key and not 'tail' in old_key:
            new_ckpt[old_key] = old_value.detach()
        else:
            if 'head' in old_key:
                c_in = new_value.shape[1]
                c_in_old = old_value.shape[1]
                if c_in == c_in_old:
                    new_ckpt[old_key] = old_value.detach()
                elif c_in < c_in_old:
                    # We copy over the image channels (e.g., c_in-1 channels)
                    # and the last channel is the noise map channel
                    new_val_temp = torch.zeros_like(new_value.detach())
                    # Last channel is noise map
                    new_val_temp[:, -1:] = old_value[:, -1:]
                    # First c_in-1 channels are image (magnitude, etc)
                    new_val_temp[:, 0:c_in-1] = old_value[:, 0:c_in-1]
                    new_ckpt[old_key] = new_val_temp
                elif c_in > c_in_old:
                    new_val_temp = torch.zeros_like(new_value.detach())
                    new_val_temp[:, :c_in_old] = old_value
                    new_ckpt[old_key] = new_val_temp

            elif 'tail' in old_key:
                c_out = new_value.shape[0]
                c_out_old = old_value.shape[0]
                if c_out == c_out_old:
                    new_ckpt[old_key] = old_value.detach()
                elif c_out < c_out_old:
                    new_ckpt[old_key] = old_value[:c_out].detach()
                elif c_out > c_out_old:
                    new_val_temp = torch.zeros_like(new_value.detach())
                    new_val_temp[:c_out_old] = old_value
                    new_ckpt[old_key] = new_val_temp
    return new_ckpt
