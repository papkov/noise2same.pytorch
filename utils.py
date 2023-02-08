import torch.nn
from torch.nn import Identity
from omegaconf import DictConfig
from typing import Tuple

from noise2same.backbone import SwinIR, UNet, RegressionHead
from noise2same.backbone.swinia import SwinIA
from noise2same.backbone.bsp_swinir import BSpSwinIR
from noise2same.dataset.getter import compute_pad_divisor


def recalculate_img_size(cfg: DictConfig) -> int:
    """
    Recalculate image size with respect to future padding
    :param cfg: DictConfig, training/evaluation configuration object
    :return: int
    """
    pad_divisor = compute_pad_divisor(cfg)
    if cfg.training.crop % pad_divisor:
        return (cfg.training.crop // pad_divisor + 1) * pad_divisor
    else:
        return cfg.training.crop


def parametrize_backbone_and_head(cfg: DictConfig) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Create backbone and head according to the configuration
    :param cfg: DictConfig, training/evaluation configuration object
    :return: Tuple[torch.nn.Module, torch.nn.Module]
    """
    head = Identity()
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
    elif cfg.backbone_name == 'swinir':
        assert cfg.data.n_dim == 2
        backbone = SwinIR(
            in_chans=cfg.data.n_channels,
            img_size=recalculate_img_size(cfg),
            **cfg.backbone
        )
    elif cfg.backbone_name == 'bsp_swinir':
        assert cfg.data.n_dim == 2
        backbone = BSpSwinIR(
            in_chans=cfg.data.n_channels,
            img_size=recalculate_img_size(cfg),
            **cfg.backbone
        )
    elif cfg.backbone_name == 'swinia':
        assert cfg.data.n_dim == 2
        backbone = SwinIA(
            in_channels=cfg.data.n_channels,
            **cfg.backbone
        )
    else:
        raise ValueError("Incorrect backbone name")
    return backbone, head
