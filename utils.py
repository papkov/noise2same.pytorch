from torch.nn import Identity
from omegaconf import DictConfig
from noise2same.swinir import SwinIR
from noise2same.unet import UNet, RegressionHead


def parametrize_backbone_and_head(cfg: DictConfig):
    if cfg.backbone_name == 'unet':
        backbone = UNet(**cfg.backbone)
        head = RegressionHead(
            in_channels=cfg.backbone.base_channels,
            out_channels=cfg.backbone.in_channels,
            n_dim=cfg.backbone.n_dim
        )
    else:
        head = Identity()
        if cfg.backbone_name == 'swinir':
            assert cfg.data.n_dim == 2
            backbone = SwinIR(**cfg.backbone)
        else:
            backbone = Identity()
    return backbone, head
