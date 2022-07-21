from torch.nn import Identity
from omegaconf import DictConfig
from noise2same.swinir import SwinIR
from noise2same.unet import UNet, RegressionHead
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
        head = Identity()
        if cfg.backbone_name == 'swinir':
            assert cfg.data.n_dim == 2
            backbone = SwinIR(
                in_chans=cfg.data.n_channels,
                img_size=recalculate_img_size(cfg),
                **cfg.backbone
            )
        else:
            backbone = Identity()
    return backbone, head
