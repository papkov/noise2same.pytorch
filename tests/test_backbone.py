import os
from pathlib import Path
from typing import List, Tuple, Union

import pytest
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import Identity

from noise2same.backbone import unet, unet_b2u, swinia, swinir
from noise2same.util import register_config_resolvers

Ints = Union[int, Tuple[int, ...]]


@pytest.fixture(scope="module", autouse=True)
def register_config_resolvers_fixture():
    register_config_resolvers()


@pytest.mark.parametrize('backbone_name,expected_backbone,expected_head',
                         [('unet', unet.UNet, unet.RegressionHead),
                          ('unet_b2u', unet_b2u.UNetB2U, Identity),
                          ('swinia', swinia.SwinIA, Identity),
                          ('swinir', swinir.SwinIR, Identity),
                          ])
def test_backbone(backbone_name: str, expected_backbone: type, expected_head: type):
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}
    with initialize(version_base=None, config_path="../config/backbone"):
        cfg = compose(config_name=backbone_name, return_hydra_config=False,
                      overrides=['+dataset.n_channels=1', '+dataset.n_dim=2', '+dataset.input_size=64',
                                 '+training.steps=5000'])
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        print('\n', OmegaConf.to_yaml(cfg))

    backbone = instantiate(cfg.backbone)
    head = instantiate(cfg.head)
    assert isinstance(backbone, expected_backbone)
    assert isinstance(head, expected_head)

    x = torch.randn(1, 1, 64, 64)
    x = backbone(x)
    x = head(x)
    assert x.shape == (1, 1, 64, 64)


@pytest.mark.parametrize('backbone_name,expected_backbone,expected_head',
                         [('unet', unet.UNet, unet.RegressionHead),
                          ('unet_b2u', unet_b2u.UNetB2U, Identity),
                          ])
def test_backbone_3d(backbone_name: str, expected_backbone: type, expected_head: type):
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}
    with initialize(version_base=None, config_path="../config/backbone"):
        cfg = compose(config_name=backbone_name, return_hydra_config=False,
                      overrides=['+dataset.n_channels=1', '+dataset.n_dim=3', '+training.steps=5000'])
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        print('\n', OmegaConf.to_yaml(cfg))

    backbone = instantiate(cfg.backbone)
    head = instantiate(cfg.head)
    assert isinstance(backbone, expected_backbone)
    assert isinstance(head, expected_head)

    x = torch.randn(1, 1, 8, 32, 32)
    x = backbone(x)
    x = head(x)
    assert x.shape == (1, 1, 8, 32, 32)


@pytest.mark.parametrize('shape,factor',
                         [((32, 32, 32), 2),
                          ((32, 32, 32), [2, 2, 2, 2, 2]),
                          # fails because of the downsampling factor is larger than the input depth
                          pytest.param((8, 32, 32), [2, 2, 2, 2, 2], marks=pytest.mark.xfail),
                          # passes because only three depth downsamplings
                          ((8, 32, 32), [(1, 2, 2), 2, (1, 2, 2), 2, (2, 2, 2)]),
                          ])
def test_factor_unet_b2u(shape: Tuple[int, ...], factor: Union[int, List[Ints]]):
    backbone = unet_b2u.UNetB2U(in_channels=1, n_dim=3, factor=factor)
    head = Identity()

    x = torch.randn(2, 1, *shape)
    x = backbone(x)
    x = head(x)
    assert x.shape == (2, 1) + shape


@pytest.mark.parametrize('downsampling', [('conv', 'conv'), ('conv', 'res'), ('res', 'res')])
@pytest.mark.parametrize('shape,factor',
                         [((4, 4, 4), 2),
                          ((4, 4, 4), [2, 2]),
                          # fails because of the downsampling factor is larger than the input depth
                          pytest.param((2, 4, 4), [2, 2], marks=pytest.mark.xfail),
                          # passes because only one depth downsampling
                          ((2, 4, 4), [(1, 2, 2), 2]),
                          ])
def test_factor_unet(downsampling: Tuple[str, ...], shape: Tuple[int, ...], factor: Union[int, List[Ints]]):
    backbone = unet.UNet(in_channels=1, n_dim=3, downsampling_factor=factor, downsampling=downsampling)
    print(backbone)
    head = unet.RegressionHead(in_channels=96, n_dim=3, out_channels=1)

    x = torch.randn(2, 1, *shape)
    x = backbone(x)
    x = head(x)
    assert x.shape == (2, 1) + shape


@pytest.mark.parametrize('window_size,dilations,shuffles,pad_divisor',
                         [(8, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 8),
                          (8, [1, 2, 4, 2, 1], [1, 1, 1, 1, 1], 32),
                          (8, [1, 1, 1, 1, 1], [1, 2, 4, 2, 1], 32),
                          (8, [1, 2, 4, 2, 1], [1, 2, 4, 2, 1], 128),
                          (8, [1, 4, 1, 1, 1], [1, 1, 1, 4, 1], 32),
                          ])
def test_swinia_pad_divisor(window_size: int, dilations: List[int], shuffles: List[int], pad_divisor: int):
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}
    with initialize(version_base=None, config_path="../config/backbone"):
        cfg = compose(config_name='swinia', return_hydra_config=False,
                      overrides=['+dataset.n_channels=1', '+dataset.n_dim=2', '+dataset_train.input_size=64',
                                 '+training.steps=5000', f'backbone.window_size={window_size}',
                                 f'backbone.dilations={dilations}', f'backbone.shuffles={shuffles}'])
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        print('\n', OmegaConf.to_yaml(cfg))

    assert cfg.backbone.pad_divisor == pad_divisor
