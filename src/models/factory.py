from .drunet import DRUNet
from .nafnet import NAFNet
from .scunet import SCUNet
from .unet import UNet

def get_model(model_name, config):
    model_name = model_name.lower()
    
    # Common args
    in_c = config['common']['in_channels']
    out_c = config['common']['out_channels']
    
    if model_name == 'drunet':
        return DRUNet(
            in_channels=in_c,
            out_channels=out_c,
            base_channels=config['drunet']['base_channels']
        )
    elif model_name in ['nafnet', 'nafnet_small', 'nafnet_medium', 'nafnet_large']:
        model_cfg = config[model_name]
        return NAFNet(
            img_channel=in_c,
            width=model_cfg['width'],
            enc_blk_nums=model_cfg['enc_blk_nums'],
            middle_blk_num=model_cfg['middle_blk_num'],
            dec_blk_nums=model_cfg['dec_blk_nums']
        )
    elif model_name == 'scunet':
        return SCUNet(
            in_channels=in_c,
            out_channels=out_c,
            config=config['scunet']['config']
        )
    elif model_name == 'unet':
        return UNet(
            n_channels=in_c,
            n_classes=out_c,
            bilinear=config['unet']['bilinear']
        )
    else:
        raise ValueError(f"Model {model_name} not implemented.")
