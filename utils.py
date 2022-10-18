from torch.nn import Identity
from omegaconf import DictConfig

from noise2same.backbone import SwinUPer, SwinIR, UNet, RegressionHead, ConvSwinUPer, SUNet
from noise2same.backbone.bsp_swinir import BSPSwinIR
from noise2same.dataset.getter import compute_pad_divisor


def recalculate_img_size(cfg: DictConfig) -> int:
    pad_divisor = compute_pad_divisor(cfg)
    if cfg.training.crop % pad_divisor:
        return (cfg.training.crop // pad_divisor + 1) * pad_divisor
    else:
        return cfg.training.crop


def parametrize_backbone_and_head(cfg: DictConfig):
    if cfg.backbone_name == 'unet':
        backbone = UNet(
            in_channels=cfg.data.n_channels,
            **cfg.backbone
        )
        head = RegressionHead(
            in_channels=cfg.backbone.base_channels,
            out_channels=cfg.data.n_channels,
            n_dim=cfg.data.n_dim
        )
    else:
        if cfg.backbone_name == 'swinir':
            assert cfg.data.n_dim == 2
            backbone = SwinIR(
                in_chans=cfg.data.n_channels,
                img_size=recalculate_img_size(cfg),
                **cfg.backbone
            )
            head = Identity()
        elif cfg.backbone_name == 'bsp_swinir':
            assert cfg.data.n_dim == 2
            backbone = BSPSwinIR(
                in_chans=cfg.data.n_channels,
                img_size=recalculate_img_size(cfg),
                **cfg.backbone
            )
            head = Identity()
        elif cfg.backbone_name == 'swin_uper':
            assert cfg.data.n_dim == 2
            backbone = SwinUPer(
                in_chans=cfg.data.n_channels,
                img_size=recalculate_img_size(cfg),
                **cfg.backbone
            )
            head = Identity()
        elif cfg.backbone_name == 'conv_swin_uper':
            assert cfg.data.n_dim == 2
            backbone = ConvSwinUPer(
                in_chans=cfg.data.n_channels,
                img_size=recalculate_img_size(cfg),
                **cfg.backbone
            )
            head = Identity()
        elif cfg.backbone_name == 'sunet':
            assert cfg.data.n_dim == 2
            backbone = SUNet(
                in_chans=cfg.data.n_channels,
                out_chans=cfg.data.n_channels,
                img_size=recalculate_img_size(cfg),
                **cfg.backbone
            )
            head = Identity()
        else:
            backbone = Identity()
    return backbone, head
