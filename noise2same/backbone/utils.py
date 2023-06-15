from typing import Tuple

import torch.nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from noise2same.dataset.getter import compute_pad_divisor


def recalculate_img_size(cfg: DictConfig) -> int:
    """
    Recalculate image size with respect to future padding
    :param cfg: DictConfig, training/evaluation configuration object
    :return: int
    """
    # TODO training.crop will be deprecated and moved to transforms config. Consider removing this function
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
    backbone = instantiate(cfg.backbone)(input_size=recalculate_img_size(cfg))
    head = instantiate(cfg.head)
    return backbone, head
