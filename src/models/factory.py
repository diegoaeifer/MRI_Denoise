import torch
import torch.nn as nn
from .drunet import DRUNet
from .nafnet import NAFNet
from .scunet import SCUNet
from .unet import UNet


class ChannelAdapter(nn.Module):
    """
    Adapts 2-channel input (noisy_image + sigma_map) to 1-channel for pretrained deepinv models.

    Strategy: A small learnable 2→1 conv fuses both channels. The pretrained backbone
    weights are NOT altered — only this tiny adapter is trained. The sigma map acts as
    noise-level conditioning that the adapter learns to incorporate into the image stream.
    """
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        # Initialize so it starts as a near-identity on the image channel
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        # Slightly activate the image channel (index 0)
        with torch.no_grad():
            self.proj[0].weight[:, 0, 1, 1] = 0.9   # favour image channel
            self.proj[0].weight[:, 1, 1, 1] = 0.1   # light sigma influence

    def forward(self, x):
        return self.proj(x)


class DeepinvPretrainedModel(nn.Module):
    """
    Wraps a deepinv pretrained 1-channel model with a ChannelAdapter for
    2-channel (image + noise_map) input pipelines. The backbone is left intact;
    only the adapter head is fine-tuned unless full fine-tuning is desired.
    """
    def __init__(self, backbone: nn.Module, in_channels: int = 2, backbone_in_channels: int = 1, freeze_backbone: bool = False):
        super().__init__()
        self.adapter = ChannelAdapter(in_channels=in_channels) if in_channels != 1 else nn.Identity()
        self.backbone = backbone
        self.backbone_in_channels = backbone_in_channels
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x is (B, 2, H, W) -> [Noisy Image, Sigma Map]
        # img = x[:, 0:1, :, :]
        sigma_map = x[:, 1:2, :, :]
        
        # Deepinv pretrained models (DRUNet, DnCNN, etc.) typically expect 
        # sigma as [B, 1, 1, 1] scalar value rather than a full map.
        sigma_scalar = sigma_map.mean(dim=(1, 2, 3)) # (B,)
        
        # We still use the adapter on the full 'x' (image + map)
        x_1ch = self.adapter(x)  # (B, 1, H, W)
        
        # Adapt to 3-channel backbone if necessary (e.g. SCUNet)
        if self.backbone_in_channels == 3:
            x_in = x_1ch.repeat(1, 3, 1, 1)
        else:
            x_in = x_1ch

        # Determine extra arguments based on backbone type
        # RAM model requires img_size if no physics operator is provided
        try:
            import deepinv
            if isinstance(self.backbone, deepinv.models.RAM):
                 # RAM expects sigma as tensor [B] or scalar
                 return self.backbone(x_in, sigma=sigma_scalar, img_size=x_in.shape[2:])
        except (ImportError, AttributeError):
            pass

        # Most deepinv models (DRUNet, DnCNN, etc.) expect (x, sigma)
        # Some models might output a single tensor, others a tuple?
        out = self.backbone(x_in, sigma_scalar)
        
        # SCUNet/DRUNet outputs might be 3-ch if weights are 3-ch
        if isinstance(out, torch.Tensor) and out.shape[1] == 3:
            out = out.mean(dim=1, keepdim=True)
            
        return out



def get_model(model_name, config):
    model_name = model_name.lower()

    # Common args
    in_c = config['common']['in_channels']   # 2 for (image + sigma_map)
    out_c = config['common']['out_channels']  # 1

    # ------------------------------------------------------------------ #
    #  Custom / from-scratch models
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  DeepInverse pretrained models (2-channel adaptation via ChannelAdapter)
    # ------------------------------------------------------------------ #
    elif model_name == 'drunet_pretrained':
        import deepinv
        backbone = deepinv.models.DRUNet(
            in_channels=1,
            out_channels=out_c,
            pretrained='download'
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'dncnn_pretrained':
        import deepinv
        backbone = deepinv.models.DnCNN(
            in_channels=1,
            out_channels=out_c,
            pretrained='download'
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'scunet_pretrained':
        import deepinv
        backbone = deepinv.models.SCUNet(
            in_nc=3,
            pretrained='download'
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c, backbone_in_channels=3)

    elif model_name == 'swinir_pretrained':
        import deepinv
        backbone = deepinv.models.SwinIR(
            in_chans=1,
            pretrained='download'
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'restormer':
        import deepinv
        pretrained_cfg = config.get('restormer', {}).get('pretrained', 'denoising')
        backbone = deepinv.models.Restormer(
            in_channels=1,
            out_channels=out_c,
            pretrained=pretrained_cfg
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'gsdrunet':
        import deepinv
        pretrained_cfg = config.get('gsdrunet', {}).get('pretrained', 'download')
        backbone = deepinv.models.GSDRUNet(
            in_channels=1,
            pretrained=pretrained_cfg
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'ram_pretrained':
        import deepinv
        # Foundation model: Reconstruct Anything Model
        # Needs in_channels as a list/sequence
        backbone = deepinv.models.RAM(
            in_channels=[1],
            pretrained=True
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'swinir':
        import deepinv
        model = deepinv.models.SwinIR(in_chans=in_c)
        return model

    else:
        raise ValueError(f"Model '{model_name}' not implemented. "
                         f"Valid options: drunet, nafnet, scunet, unet, "
                         f"drunet_pretrained, dncnn_pretrained, scunet_pretrained, "
                         f"swinir_pretrained, restormer, gsdrunet, swinir")

