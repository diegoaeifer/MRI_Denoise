import torch
from .drunet import DRUNet
from .nafnet import NAFNet
from .scunet import SCUNet
from .unet import UNet

def get_model(model_name, config):
    model_name = model_name.lower()
    
    # Common args
    in_c = config['common']['in_channels']
    out_c = config['common']['out_channels']
    
    class DeepinvWrapper(torch.nn.Module):
        def __init__(self, model_class, pretrained_name, **kwargs):
            super().__init__()
            import deepinv
            # Handle deepinv expected '1' channel geometry natively
            self.input_align = torch.nn.Conv2d(in_c, 1, kernel_size=1) if in_c != 1 else torch.nn.Identity()
            
            # Special case for models that don't support the 'pretrained' keyword in __init__ consistently
            if model_name == 'gsdrunet':
                self.model = model_class(in_channels=1, pretrained=pretrained_name, **kwargs)
            elif model_name == 'restormer':
                self.model = model_class(in_channels=1, pretrained=pretrained_name, **kwargs)
            else:
                self.model = model_class(in_channels=1, **kwargs)
                
            if pretrained_name:
                print(f"Deepinv '{model_name}' weights initialized.")
                
        def forward(self, x):
            # Ensure input_align is on the same device
            if hasattr(self.input_align, 'weight') and self.input_align.weight.device != x.device:
                self.input_align = self.input_align.to(x.device)
            x = self.input_align(x)
            return self.model(x)
    
    if model_name == 'drunet':
        return DRUNet(
            in_channels=in_c,
            out_channels=out_c,
            base_channels=config['drunet']['base_channels']
        )
    elif model_name in ['nafnet', 'nafnet_xs', 'nafnet_small', 'nafnet_medium', 'nafnet_large']:
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
    elif model_name == 'restormer':
        import deepinv
        pretrained_cfg = config.get('restormer', {}).get('pretrained', "denoising")
        return DeepinvWrapper(deepinv.models.Restormer, pretrained_cfg, out_channels=out_c)
        
    elif model_name == 'gsdrunet':
        import deepinv
        pretrained_cfg = config.get('gsdrunet', {}).get('pretrained', "download")
        return DeepinvWrapper(deepinv.models.GSDRUNet, pretrained_cfg, out_channels=out_c)
        
    elif model_name == 'swinir':
        import deepinv
        # SwinIR in deepinv expects in_chans
        model = deepinv.models.SwinIR(in_chans=in_c)
        return model
    else:
        raise ValueError(f"Model {model_name} not implemented.")
