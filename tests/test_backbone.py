import os
from pathlib import Path

import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import Identity

from noise2same.backbone import unet, swinia, swinir
from noise2same.backbone.utils import recalculate_img_size


@pytest.mark.parametrize('backbone_name,expected_backbone,expected_head',
                         [('unet', unet.UNet, unet.RegressionHead),
                          ('swinia', swinia.SwinIA, Identity),
                          ('swinir', swinir.SwinIR, Identity),
                          ])
def test_backbone(backbone_name, expected_backbone, expected_head):
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}
    with initialize(version_base=None, config_path="../config/backbone"):
        cfg = compose(config_name=backbone_name, return_hydra_config=False,
                      overrides=['+dataset.n_channels=1', '+dataset.n_dim=2', '+training.crop=64'])
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        print('\n', OmegaConf.to_yaml(cfg))

    backbone = instantiate(cfg.backbone)(input_size=recalculate_img_size(cfg))
    head = instantiate(cfg.head)
    assert isinstance(backbone, expected_backbone)
    assert isinstance(head, expected_head)
