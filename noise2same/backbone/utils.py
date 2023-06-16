from typing import Tuple

import torch.nn
from hydra.utils import instantiate
from omegaconf import DictConfig


def parametrize_backbone_and_head(cfg: DictConfig) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Create backbone and head according to the configuration
    :param cfg: DictConfig, training/evaluation configuration object
    :return: Tuple[torch.nn.Module, torch.nn.Module]
    """
    backbone = instantiate(cfg.backbone)
    head = instantiate(cfg.head)
    return backbone, head
