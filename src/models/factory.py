import torch
import torch.nn as nn
from .drunet import DRUNet
from .nafnet import NAFNet
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
        
        # Adapt to 3-channel backbone if necessary
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
        
        # Collapse 3-ch output back to 1-ch if backbone used 3-channel weights
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


    elif model_name == '3d-parallel-ricianet':
        from .rician_net3d import RicianNet3D
        return RicianNet3D(
            in_channels=in_c,
            out_channels=out_c,
            base_filters=config.get('3d-parallel-ricianet', {}).get('base_filters', 16)
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

    elif model_name == 'unet_pretrained':
        import deepinv
        backbone = deepinv.models.UNet(
            in_channels=1,
            out_channels=out_c
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'drunet_pretrained':
        import deepinv
        backbone = deepinv.models.DRUNet(
            in_channels=1,
            out_channels=out_c,
            pretrained='download'
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

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

    elif model_name == 'bm3d':
        import deepinv
        # Unsupervised, standard BM3D wrapping
        # Deepinv BM3D doesn't need pretraining
        backbone = deepinv.models.BM3D()
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'dip':
        import deepinv
        # Unsupervised, Deep Image Prior. Note DIP is typically optimized per image.
        # This will return the architecture (e.g., UNet) typically used for DIP.
        backbone = deepinv.models.UNet(in_channels=1, out_channels=1)
        # Deepinv provides DIP typically as an optimization algorithm, but we can provide the backbone.
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == 'imt-mrd':
        from .imt_mrd_wrapper import ImtMrdWrapper  # noqa: PLC0415
        cfg = config.get('imt_mrd', {})
        return ImtMrdWrapper(
            model_path=cfg.get('model_path', None),
            freeze_backbone=cfg.get('freeze_backbone', True),
            model_variant=cfg.get('model_variant', 'complex'),
        )

    elif model_name == 'snraware':
        from .snraware_wrapper import SNRAwareWrapper  # noqa: PLC0415
        cfg = config.get('snraware', {})
        return SNRAwareWrapper(
            model_path=cfg.get('model_path', None),
            model_size=cfg.get('model_size', 'medium'),
            overlap=cfg.get('overlap', 32),
            freeze=cfg.get('freeze', True),
            use_sigma_as_gmap=cfg.get('use_sigma_as_gmap', False),
        )

    elif model_name == 'cdlnet':
        from .cdlnet_wrapper import CDLNetWrapper  # noqa: PLC0415
        cfg = config.get('cdlnet', {})
        return CDLNetWrapper(
            K=cfg.get('K', 30),
            M=cfg.get('M', 169),
            P=cfg.get('P', 7),
            s=cfg.get('s', 2),
            adaptive=cfg.get('adaptive', True),
            init=cfg.get('init', False),
            weights_path=cfg.get('weights_path', None),
        )

    elif model_name == 'restore_rwkv':
        from .restore_rwkv_wrapper import RestoreRWKVWrapper  # noqa: PLC0415
        cfg = config.get('restore_rwkv', {})
        return RestoreRWKVWrapper(
            dim=cfg.get('dim', 48),
            num_blocks=cfg.get('num_blocks', [4, 6, 6, 8]),
            num_refinement_blocks=cfg.get('num_refinement_blocks', 4),
        )

    elif model_name == 'astro_denoiser':
        from .astro_denoiser_wrapper import AstroDenoiserWrapper  # noqa: PLC0415
        cfg = config.get('astro_denoiser', {})
        return AstroDenoiserWrapper(
            filters=cfg.get('filters', 32),
            depth=cfg.get('depth', 6),
        )

    elif model_name == 'nlmced':
        from .nlmced_wrapper import NLmCEDWrapper  # noqa: PLC0415
        cfg = config.get('nlmced', {})
        return NLmCEDWrapper(
            mode=cfg.get('mode', 'auto'),
            iterations=cfg.get('iterations', 1),
            rho=cfg.get('rho', 0.01),
            alpha=cfg.get('alpha', 0.01),
            num=cfg.get('num', 1),
        )

    elif model_name in ('foura_nafnet', 'foura_nafnet_small'):
        from .foura_adapter import FouRAWrapper  # noqa: PLC0415
        base = NAFNet(
            img_channel=config.get('in_channels', in_c),
            width=config.get('width', 64),
        )
        weights = config.get('pretrained_weights')
        if weights:
            base.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        return FouRAWrapper(
            base,
            rank=config.get('rank', 16),
            alpha=config.get('alpha', 32.0),
        )

    elif model_name in ('foura_drunet',):
        from .foura_adapter import FouRAWrapper  # noqa: PLC0415
        base = DRUNet(
            in_channels=config.get('in_channels', in_c),
            out_channels=out_c,
            base_channels=config.get('base_channels', 64),
        )
        weights = config.get('pretrained_weights')
        if weights:
            base.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        return FouRAWrapper(
            base,
            rank=config.get('rank', 16),
            alpha=config.get('alpha', 32.0),
        )

    else:
        raise ValueError(f"Model '{model_name}' not implemented. "
                         f"Valid options: drunet, nafnet, unet, "
                         f"drunet_pretrained, restormer, gsdrunet, ram_pretrained, bm3d, dip, "
                         f"imt-mrd, snraware, cdlnet, restore_rwkv, astro_denoiser, nlmced, "
                         f"foura_nafnet, foura_nafnet_small, foura_drunet")

