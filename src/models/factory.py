import torch
import torch.nn as nn
from .drunet import DRUNet
from .nafnet import NAFNet
from .scunet import SCUNet
from .unet import UNet


class ChannelAdapter(nn.Module):
    """
    Adapts 2-channel input (noisy_image + sigma_map) to 1-channel
    for pretrained deepinv models.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1))
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        with torch.no_grad():
            self.proj[0].weight[:, 0, 1, 1] = 0.9
            self.proj[0].weight[:, 1, 1, 1] = 0.1

    def forward(self, x):
        return self.proj(x)


class DeepinvPretrainedModel(nn.Module):
    """
    Wraps a deepinv pretrained 1-channel model with a ChannelAdapter for
    2-channel (image + noise_map) input pipelines. The backbone is left intact;
    only the adapter head is fine-tuned unless full fine-tuning is desired.
    """

    def __init__(self,
                 backbone: nn.Module,
                 in_channels: int = 2,
                 backbone_in_channels: int = 1,
                 freeze_backbone: bool = False):
        super().__init__()
        self.adapter = ChannelAdapter(
            in_channels=in_channels) if in_channels != 1 else nn.Identity()
        self.backbone = backbone
        self.backbone_in_channels = backbone_in_channels
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x is (B, 2, H, W) -> [Noisy Image, Sigma Map]
        sigma_map = x[:, 1:2, :, :]
        sigma_scalar = sigma_map.mean(dim=(1, 2, 3))  # (B,)

        x_1ch = self.adapter(x)  # (B, 1, H, W)

        if self.backbone_in_channels == 3:
            x_in = x_1ch.repeat(1, 3, 1, 1)
        else:
            x_in = x_1ch

        try:
            import deepinv
            if isinstance(self.backbone, deepinv.models.RAM):
                return self.backbone(x_in,
                                     sigma=sigma_scalar,
                                     img_size=x_in.shape[2:])
        except (ImportError, AttributeError):
            pass

        out = self.backbone(x_in, sigma_scalar)

        if isinstance(out, torch.Tensor) and out.shape[1] == 3:
            out = out.mean(dim=1, keepdim=True)

        return out


class SNRAwareWrapper(nn.Module):
    """
    Wrapper for SNRAware pretrained model.
    The SNRAware model expects [B, 3, T, H, W] for real/imag + g-factor map.
    Since we only have magnitude images and sigma map, we format our
    input as [B, 3, T, H, W] by passing [Magnitude, Zeros, Sigma].
    """
    def __init__(self, in_channels=2, model_path="src/models/snraware_small_model.pts", freeze_backbone=False):
        super().__init__()
        import os
        # Path resolution: assuming factory.py is in src/models
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, os.path.basename(model_path))

        # load the torchscript model
        self.model = torch.jit.load(model_path, map_location='cpu')

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        # The training factory passes (B, 2, H, W) or (B, 2, D, H, W) mostly.
        # Check dimensionality
        is_2d = x.dim() == 4

        if is_2d:
            # Input: (B, 2, H, W) -> output (B, 1, H, W) expected.
            # But the underlying torchscript model expects [B, 3, T, H, W] where T=1
            B, C, H, W = x.shape
            magnitude = x[:, 0:1, :, :]
            sigma = x[:, 1:2, :, :]
            zeros = torch.zeros_like(magnitude)

            # Form [B, 3, T=1, H, W]
            x_in = torch.cat([magnitude.unsqueeze(2), zeros.unsqueeze(2), sigma.unsqueeze(2)], dim=1)

            # Forward pass through SNRAware
            out = self.model(x_in) # (B, 2, T=1, H, W) for real/imag

            # Calculate magnitude
            out_real = out[:, 0, 0, :, :]
            out_imag = out[:, 1, 0, :, :]
            out_mag = torch.sqrt(out_real**2 + out_imag**2).unsqueeze(1) # (B, 1, H, W)
            return out_mag
        else:
            # Input: (B, 2, D, H, W)
            B, C, D, H, W = x.shape
            magnitude = x[:, 0:1, :, :, :]
            sigma = x[:, 1:2, :, :, :]
            zeros = torch.zeros_like(magnitude)

            # Form [B, 3, D, H, W]
            x_in = torch.cat([magnitude, zeros, sigma], dim=1)

            # Forward pass
            out = self.model(x_in) # (B, 2, D, H, W)

            # Calculate magnitude
            out_real = out[:, 0, :, :, :]
            out_imag = out[:, 1, :, :, :]
            out_mag = torch.sqrt(out_real**2 + out_imag**2).unsqueeze(1) # (B, 1, D, H, W)
            return out_mag

def _build_drunet(in_c, out_c, config, model_name):
    return DRUNet(in_channels=in_c,
                  out_channels=out_c,
                  base_channels=config['drunet']['base_channels'])


def _build_nafnet(in_c, out_c, config, model_name):
    model_cfg = config[model_name]
    return NAFNet(img_channel=in_c,
                  width=model_cfg['width'],
                  enc_blk_nums=model_cfg['enc_blk_nums'],
                  middle_blk_num=model_cfg['middle_blk_num'],
                  dec_blk_nums=model_cfg['dec_blk_nums'])


def _build_se_scunet_mini(in_c, out_c, config, model_name):
    return SE_SCUNet_mini(in_nc=in_c,
                          config=config.get('se_scunet_mini',
                                            {}).get('config',
                                                    [1, 1, 1, 1, 1, 1, 1]),
                          dim=config.get('se_scunet_mini', {}).get('dim', 64))


def _build_scunet(in_c, out_c, config, model_name):
    return SCUNet(in_channels=in_c,
                  out_channels=out_c,
                  config=config['scunet']['config'])


def _build_unet(in_c, out_c, config, model_name):
    return UNet(n_channels=in_c,
                n_classes=out_c,
                bilinear=config['unet']['bilinear'])


def _build_ffdnet(in_c, out_c, config, model_name):
    return FFDNet(in_nc=in_c,
                  out_nc=out_c,
                  nc=config.get('ffdnet', {}).get('nc', 64),
                  nb=config.get('ffdnet', {}).get('nb', 15))


def _build_unet_pretrained(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.UNet(in_channels=1, out_channels=out_c)
    return DeepinvPretrainedModel(backbone, in_channels=in_c)


def _build_drunet_pretrained(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.DRUNet(in_channels=1,
                                     out_channels=out_c,
                                     pretrained='download')
    return DeepinvPretrainedModel(backbone, in_channels=in_c)


def _build_dncnn_pretrained(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.DnCNN(in_channels=1,
                                    out_channels=out_c,
                                    pretrained='download')
    return DeepinvPretrainedModel(backbone, in_channels=in_c)


def _build_scunet_pretrained(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.SCUNet(in_nc=3, pretrained='download')
    return DeepinvPretrainedModel(backbone,
                                  in_channels=in_c,
                                  backbone_in_channels=3)


def _build_swinir_pretrained(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.SwinIR(in_chans=1, pretrained='download')
    return DeepinvPretrainedModel(backbone,
                                  in_channels=in_c,
                                  backbone_in_channels=1)


def _build_swinir(in_c, out_c, config, model_name):
    import deepinv
    return deepinv.models.SwinIR(in_chans=in_c)


def _build_restormer(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.Restormer(in_channels=1,
                                        out_channels=1,
                                        pretrained='denoising_gray')
    return DeepinvPretrainedModel(backbone,
                                  in_channels=in_c,
                                  backbone_in_channels=1)


def _build_gsdrunet(in_c, out_c, config, model_name):
    import deepinv
    pretrained_cfg = config.get('gsdrunet', {}).get('pretrained', 'download')
    backbone = deepinv.models.GSDRUNet(in_channels=1,
                                       pretrained=pretrained_cfg)
    return DeepinvPretrainedModel(backbone, in_channels=in_c)


def _build_ram_pretrained(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.RAM(in_channels=[1], pretrained=True)
    return DeepinvPretrainedModel(backbone, in_channels=in_c)


def _build_bm3d(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.BM3D()
    return DeepinvPretrainedModel(backbone, in_channels=in_c)


def _build_dip(in_c, out_c, config, model_name):
    import deepinv
    backbone = deepinv.models.UNet(in_channels=1, out_channels=1)
    return DeepinvPretrainedModel(backbone, in_channels=in_c)

def _build_snraware(in_c, out_c, config, model_name):
    pretrained_cfg = config.get('snraware', {}).get('pretrained', 'snraware_small_model.pts')
    freeze = config.get('snraware', {}).get('freeze', False)
    return SNRAwareWrapper(in_channels=in_c, model_path=pretrained_cfg, freeze_backbone=freeze)


_MODEL_BUILDERS = {
    'drunet': _build_drunet,
    'nafnet': _build_nafnet,
    'nafnet_xs': _build_nafnet,
    'nafnet_small': _build_nafnet,
    'nafnet_medium': _build_nafnet,
    'nafnet_large': _build_nafnet,
    'se_scunet_mini': _build_se_scunet_mini,
    'scunet': _build_scunet,
    'unet': _build_unet,
    'ffdnet': _build_ffdnet,
    'unet_pretrained': _build_unet_pretrained,
    'drunet_pretrained': _build_drunet_pretrained,
    'dncnn_pretrained': _build_dncnn_pretrained,
    'scunet_pretrained': _build_scunet_pretrained,
    'swinir_pretrained': _build_swinir_pretrained,
    'swinir': _build_swinir,
    'restormer': _build_restormer,
    'gsdrunet': _build_gsdrunet,
    'ram_pretrained': _build_ram_pretrained,
    'bm3d': _build_bm3d,
    'dip': _build_dip,
    'snraware': _build_snraware,
}


def get_model(model_name, config):
    model_name = model_name.lower()

    # Common args
    in_c = config['common']['in_channels']  # 2 for (image + sigma_map)
    out_c = config['common']['out_channels']  # 1

<<<<<<< HEAD
    builder = _MODEL_BUILDERS.get(model_name)
    if builder is None:
        valid_options = ", ".join(sorted(_MODEL_BUILDERS.keys()))
=======
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
        # SwinIR supports native 1-channel pretrained weights
        backbone = deepinv.models.SwinIR(
            in_chans=1,
            pretrained='download'
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c, backbone_in_channels=1)
    
    elif model_name == 'swinir':
        import deepinv
        # From scratch case should also use correct input channels (2 for image+sigma)
        return deepinv.models.SwinIR(in_chans=in_c)

    elif model_name == 'restormer':
        import deepinv
        # Restormer has a native 'denoising_gray' pretrained model
        backbone = deepinv.models.Restormer(
            in_channels=1,
            out_channels=1,
            pretrained='denoising_gray'
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c, backbone_in_channels=1)

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

    else:
>>>>>>> origin/perf-optimize-folder-processing-16091802553344257750
        raise ValueError(f"Model '{model_name}' not implemented. "
                         f"Valid options: {valid_options}")

    return builder(in_c, out_c, config, model_name)
