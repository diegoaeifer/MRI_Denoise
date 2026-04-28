import torch.nn as nn
from monai.networks.nets import BasicUNet, AttentionUnet, DynUNet

from .nafnet import NAFNet
from .drunet import DRUNet
from .scunet import SCUNet
from .visnet import DPN as VisNet
from .se_scunet_mini import SCUNet as SE_SCUNet_mini
from .rician_net3d import RicianNet3D


class TwoChannelAdapter(nn.Module):
    """
    Adapter to allow single-channel backbones (like pretrained U-Nets)
    to accept 2-channel input (image + noise map).
    """

    def __init__(self, backbone: nn.Module, spatial_dims: int = 2):
        super().__init__()
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.fuse = Conv(2, 1, kernel_size=1)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(self.fuse(x))


REGISTRY = {
    "basic_unet": BasicUNet,
    "attention_unet": AttentionUnet,
    "dynunet": DynUNet,
    "nafnet": NAFNet,
    "drunet": DRUNet,
    "scunet": SCUNet,
    "visnet": VisNet,
    "se_scunet_mini": SE_SCUNet_mini,
    "ricianet3d": RicianNet3D,
}


def get_network(name: str, **kwargs) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"Unknown network: {name}. Available: {list(REGISTRY.keys())}")

    return REGISTRY[name](**kwargs)
