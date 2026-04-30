import torch
import torch.nn as nn
from .drunet import DRUNet
from .nafnet import NAFNet
from .scunet import SCUNet
from .unet import UNet
from .ffdnet import FFDNet
from .se_scunet_mini import SCUNet as SE_SCUNet_mini
from .visnet import DPN as VisNet
from .snraware_adapter import SNRAwareAdapter


class ChannelAdapter(nn.Module):
    """
    Adapts 2-channel input (noisy_image + sigma_map) to 1-channel for pretrained models.
    Supports both 2D and 3D inputs dynamically.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.proj2d = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.proj3d = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )
        # Initialize 2D
        nn.init.zeros_(self.proj2d[0].weight)
        nn.init.zeros_(self.proj2d[0].bias)
        nn.init.zeros_(self.proj2d[2].weight)
        nn.init.zeros_(self.proj2d[2].bias)
        with torch.no_grad():
            self.proj2d[0].weight[:, 0, 1, 1] = 0.9
            self.proj2d[0].weight[:, 1, 1, 1] = 0.1
        # Initialize 3D
        nn.init.zeros_(self.proj3d[0].weight)
        nn.init.zeros_(self.proj3d[0].bias)
        nn.init.zeros_(self.proj3d[2].weight)
        nn.init.zeros_(self.proj3d[2].bias)
        with torch.no_grad():
            self.proj3d[0].weight[:, 0, 1, 1, 1] = 0.9
            self.proj3d[0].weight[:, 1, 1, 1, 1] = 0.1

    def forward(self, x):
        if x.ndim == 4:
            return self.proj2d(x)
        elif x.ndim == 5:
            return self.proj3d(x)
        else:
            raise ValueError(
                f"Unsupported input dimension for ChannelAdapter: {x.ndim}"
            )


class DeepinvPretrainedModel(nn.Module):
    """
    Wraps a deepinv pretrained 1-channel model with a ChannelAdapter for
    2-channel (image + noise_map) input pipelines. The backbone is left intact;
    only the adapter head is fine-tuned unless full fine-tuning is desired.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int = 2,
        backbone_in_channels: int = 1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.adapter = (
            ChannelAdapter(in_channels=in_channels)
            if in_channels != 1
            else nn.Identity()
        )
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
        sigma_scalar = sigma_map.mean(dim=(1, 2, 3))  # (B,)

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


class MonaiPretrainedModel(nn.Module):
    """
    Wraps a MONAI pretrained model with a ChannelAdapter for
    2-channel (image + noise_map) input pipelines.
    Pads depth dimension to a multiple of 32 to support deep downsampling
    requirements in models when Z=6 using constant padding.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int = 2,
        backbone_in_channels: int = 1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.adapter = (
            ChannelAdapter(in_channels=in_channels)
            if in_channels != 1
            else nn.Identity()
        )
        self.backbone = backbone
        self.backbone_in_channels = backbone_in_channels
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        x_1ch = self.adapter(x)

        if self.backbone_in_channels > 1:
            if x_1ch.ndim == 4:
                x_in = x_1ch.repeat(1, self.backbone_in_channels, 1, 1)
            elif x_1ch.ndim == 5:
                x_in = x_1ch.repeat(1, self.backbone_in_channels, 1, 1, 1)
        else:
            x_in = x_1ch

        original_d = None
        if x_in.ndim == 5:
            d = x_in.shape[4]
            if d % 32 != 0:

                original_d = d
                pad_dims = []
                for dim_idx in range(4, 1, -1):
                    dim_size = x_in.shape[dim_idx]
                    rem = dim_size % 32
                    if rem != 0:
                        pad_dims.extend([0, 32 - rem])
                    else:
                        pad_dims.extend([0, 0])

                import torch.nn.functional as F

                x_in = F.pad(x_in, tuple(pad_dims), mode="constant", value=0.0)

        out = self.backbone(x_in)

        if isinstance(out, tuple):
            out = out[0]

        if isinstance(out, torch.Tensor) and out.shape[1] > 1:
            out = out.mean(dim=1, keepdim=True)

        if original_d is not None and out.ndim == 5:
            out = out[:, :, : x.shape[2], : x.shape[3], : x.shape[4]]

        return out


def get_model(model_name, config):
    model_name = model_name.lower()

    # Common args
    in_c = config["common"]["in_channels"]  # 2 for (image + sigma_map)
    out_c = config["common"]["out_channels"]  # 1

    # ------------------------------------------------------------------ #
    #  Custom / from-scratch models
    # ------------------------------------------------------------------ #
    if model_name == "drunet":
        return DRUNet(
            in_channels=in_c,
            out_channels=out_c,
            base_channels=config["drunet"]["base_channels"],
        )

    elif model_name == "visnet":
        return VisNet(in_channels=in_c, out_channels=out_c)
    elif model_name in [
        "nafnet",
        "nafnet_xs",
        "nafnet_small",
        "nafnet_medium",
        "nafnet_large",
    ]:
        model_cfg = config[model_name]
        return NAFNet(
            img_channel=in_c,
            width=model_cfg["width"],
            enc_blk_nums=model_cfg["enc_blk_nums"],
            middle_blk_num=model_cfg["middle_blk_num"],
            dec_blk_nums=model_cfg["dec_blk_nums"],
        )

    elif model_name == "se_scunet_mini":
        return SE_SCUNet_mini(
            in_nc=in_c,
            out_nc=out_c,
            config=config.get("se_scunet_mini", {}).get(
                "config", [1, 1, 1, 1, 1, 1, 1]
            ),
            dim=config.get("se_scunet_mini", {}).get("dim", 64),
        )

    elif model_name == "3d-parallel-ricianet":
        from .rician_net3d import RicianNet3D

        return RicianNet3D(
            in_channels=in_c,
            out_channels=out_c,
            base_filters=config.get("3d-parallel-ricianet", {}).get("base_filters", 16),
        )

    elif model_name == "scunet":
        return SCUNet(
            in_channels=in_c, out_channels=out_c, config=config["scunet"]["config"]
        )

    elif model_name == "snraware":
        # Optional parameters depending on your dataset dimensions
        D = config.get("dataset", {}).get("patch_depth", 16)
        H = config.get("dataset", {}).get("patch_height", 64)
        W = config.get("dataset", {}).get("patch_width", 64)
        snraware_cfg = config.get("snraware", {}).get("config", None)
        return SNRAwareAdapter(config=snraware_cfg, D=D, H=H, W=W)

    elif model_name == "unet":
        return UNet(
            n_channels=in_c, n_classes=out_c, bilinear=config["unet"]["bilinear"]
        )

    elif model_name == "ffdnet":
        return FFDNet(
            in_nc=in_c,
            out_nc=out_c,
            nc=config.get("ffdnet", {}).get("nc", 64),
            nb=config.get("ffdnet", {}).get("nb", 15),
        )

    # ------------------------------------------------------------------ #
    #  DeepInverse pretrained models (2-channel adaptation via ChannelAdapter)
    # ------------------------------------------------------------------ #

    elif model_name == "unet_pretrained":
        import deepinv

        backbone = deepinv.models.UNet(in_channels=1, out_channels=out_c)
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "drunet_pretrained":
        import deepinv

        backbone = deepinv.models.DRUNet(
            in_channels=1, out_channels=out_c, pretrained="download"
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "dncnn_pretrained":
        import deepinv

        backbone = deepinv.models.DnCNN(
            in_channels=1, out_channels=out_c, pretrained="download"
        )
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "scunet_pretrained":
        import deepinv

        backbone = deepinv.models.SCUNet(in_nc=3, pretrained="download")
        return DeepinvPretrainedModel(
            backbone, in_channels=in_c, backbone_in_channels=3
        )

    elif model_name == "swinir_pretrained":
        import deepinv

        # SwinIR supports native 1-channel pretrained weights
        backbone = deepinv.models.SwinIR(in_chans=1, pretrained="download")
        return DeepinvPretrainedModel(
            backbone, in_channels=in_c, backbone_in_channels=1
        )

    elif model_name == "swinir":
        import deepinv

        # From scratch case should also use correct input channels (2 for image+sigma)
        return deepinv.models.SwinIR(in_chans=in_c)

    elif model_name == "restormer":
        import deepinv

        # Restormer has a native 'denoising_gray' pretrained model
        backbone = deepinv.models.Restormer(
            in_channels=1, out_channels=1, pretrained="denoising_gray"
        )
        return DeepinvPretrainedModel(
            backbone, in_channels=in_c, backbone_in_channels=1
        )

    elif model_name == "gsdrunet":
        import deepinv

        pretrained_cfg = config.get("gsdrunet", {}).get("pretrained", "download")
        backbone = deepinv.models.GSDRUNet(in_channels=1, pretrained=pretrained_cfg)
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "ram_pretrained":
        import deepinv

        # Foundation model: Reconstruct Anything Model
        # Needs in_channels as a list/sequence
        backbone = deepinv.models.RAM(in_channels=[1], pretrained=True)
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "bm3d":
        import deepinv

        # Unsupervised, standard BM3D wrapping
        # Deepinv BM3D doesn't need pretraining
        backbone = deepinv.models.BM3D()
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "dip":
        import deepinv

        # Unsupervised, Deep Image Prior. Note DIP is typically optimized per image.
        # This will return the architecture (e.g., UNet) typically used for DIP.
        backbone = deepinv.models.UNet(in_channels=1, out_channels=1)
        # Deepinv provides DIP typically as an optimization algorithm, but we can provide the backbone.
        return DeepinvPretrainedModel(backbone, in_channels=in_c)

    # ------------------------------------------------------------------ #
    #  MONAI Pretrained models (2-channel adaptation via ChannelAdapter)
    # ------------------------------------------------------------------ #

    elif model_name == "monai_highresnet":
        from monai.networks.nets import HighResNet

        spatial_dims = 3 if config.get("is_3d", False) else 2
        backbone = HighResNet(
            spatial_dims=spatial_dims, in_channels=1, out_channels=out_c
        )
        return MonaiPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "monai_flexible_unet":
        from monai.networks.nets import FlexibleUNet

        spatial_dims = 3 if config.get("is_3d", False) else 2
        backbone = FlexibleUNet(
            in_channels=1,
            out_channels=out_c,
            backbone="resnet50",
            pretrained=True,
            spatial_dims=spatial_dims,
        )
        return MonaiPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "monai_resnet50":
        from monai.networks.nets import resnet50

        spatial_dims = 3 if config.get("is_3d", False) else 2
        backbone = resnet50(
            pretrained=True,
            spatial_dims=spatial_dims,
            n_input_channels=1,
            num_classes=out_c,
        )
        return MonaiPretrainedModel(backbone, in_channels=in_c)

    elif model_name == "monai_drunet":
        from .drunet import DRUNet

        spatial_dims = 3 if config.get("is_3d", False) else 2
        backbone = DRUNet(
            in_channels=1,
            out_channels=out_c,
            base_channels=config.get("drunet", {}).get("base_channels", 64),
        )
        return MonaiPretrainedModel(backbone, in_channels=in_c)

    else:
        raise ValueError(
            f"Model '{model_name}' not implemented. "
            f"Valid options: drunet, nafnet, scunet, unet, "
            f"drunet_pretrained, dncnn_pretrained, scunet_pretrained, "
            f"swinir_pretrained, restormer, gsdrunet, swinir"
        )
