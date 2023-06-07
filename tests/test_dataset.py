from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from albumentations import PadIfNeeded
from hydra.utils import instantiate
from omegaconf import OmegaConf

from noise2same.dataset import *
from noise2same.dataset.getter import get_dataset
from noise2same.dataset.util import mask_like_image
from noise2same.util import crop_as


@pytest.mark.parametrize("divisor", (2, 4, 8, 16, 32, 64))
def test_crop_as(divisor: int):
    pad = PadIfNeeded(
        min_height=None,
        min_width=None,
        pad_height_divisor=divisor,
        pad_width_divisor=divisor,
    )

    image = np.random.uniform(size=(180, 180, 1))
    padded = pad(image=image)["image"]
    cropped = crop_as(padded, image)
    print(padded.shape, cropped.shape)
    assert cropped.shape == image.shape
    assert np.all(cropped == image)


@pytest.mark.parametrize("mask_percentage", (0.1, 0.5))
def test_mask_2d(mask_percentage: float):
    shape = (64, 64, 3)
    img = np.zeros(shape)
    mask = mask_like_image(img, mask_percentage=mask_percentage, channels_last=True)
    result = mask.mean() * 100
    assert np.isclose(mask_percentage, result, atol=0.1)


@pytest.mark.parametrize("mask_percentage", (0.1, 0.5))
def test_mask_3d(mask_percentage: float):
    shape = (1, 16, 64, 64)
    img = np.zeros(shape)
    mask = mask_like_image(
        img, mask_percentage=mask_percentage, channels_last=False
    )
    result = mask.mean() * 100
    assert np.isclose(mask_percentage, result, atol=0.1)


@pytest.mark.parametrize('dataset_name,expected_dataclass,expected_dataclass_valid',
                         [('bsd68', BSD68Dataset, BSD68Dataset),
                          # ('hanzi', HanziDataset, HanziDataset), # TODO fix memory issue
                          ('imagenet', ImagenetDataset, ImagenetDataset),
                          ('microtubules', MicrotubulesDataset, None),
                          ('microtubules_generated', MicrotubulesDataset, None),
                          ('fmd', FMDDataset, FMDDataset),
                          ('fmd_deconvolution', FMDDataset, FMDDataset),
                          # ('planaria', PlanariaDataset, PlanariaDataset), # TODO fix memory issue
                          # ('sidd', SIDDDataset, SIDDDataset), # TODO move dataset
                          ('synthetic', ImagenetSyntheticDataset, Set14SyntheticDataset),
                          ('synthetic_grayscale', BSD400SyntheticDataset, BSD68SyntheticDataset),
                          ('ssi', SSIDataset, None),
                          ])
def test_dataset_instantiation(dataset_name: str, expected_dataclass: type, expected_dataclass_valid: Optional[type]):
    cfg = OmegaConf.load(f'../config/experiment/{dataset_name}.yaml')
    pad_divisor = 32

    if 'dataset_valid' in cfg:
        # Validation dataset config updates fields of the training dataset config
        cfg.dataset_valid = OmegaConf.merge(cfg.dataset, cfg.dataset_valid)
        cfg.dataset_valid.path = '../' + cfg.dataset_valid.path
        dataset_valid = instantiate(cfg.dataset_valid, pad_divisor=pad_divisor)
        assert isinstance(dataset_valid, expected_dataclass_valid)

    cfg.dataset.path = '../' + cfg.dataset.path
    if 'cached' in cfg.dataset:
        # Do not use cache for testing because of memory issues
        cfg.dataset.cached = ''
    dataset = instantiate(cfg.dataset, pad_divisor=pad_divisor)
    assert isinstance(dataset, expected_dataclass)


@pytest.mark.parametrize('dataset_name,expected_dataclass,expected_dataclass_valid',
                         [('bsd68', BSD68Dataset, BSD68Dataset),
                          # ('hanzi', HanziDataset, HanziDataset), # TODO fix memory issue
                          ('imagenet', ImagenetDataset, ImagenetDataset),
                          ('microtubules', MicrotubulesDataset, None),
                          ('microtubules_generated', MicrotubulesDataset, None),
                          ('fmd', FMDDataset, FMDDataset),
                          ('fmd_deconvolution', FMDDataset, FMDDataset),
                          # ('planaria', PlanariaDataset, PlanariaDataset), # TODO fix memory issue
                          # ('sidd', SIDDDataset, SIDDDataset), # TODO move dataset
                          ('synthetic', ImagenetSyntheticDataset, Set14SyntheticDataset),
                          ('synthetic_grayscale', BSD400SyntheticDataset, BSD68SyntheticDataset),
                          ('ssi', SSIDataset, None),
                          ])
def test_get_dataset(dataset_name: str, expected_dataclass: type, expected_dataclass_valid: Optional[type]):
    cwd = Path('..')
    cfg = OmegaConf.load(f'../config/experiment/{dataset_name}.yaml')
    cfg.update({'backbone_name': 'unet'})
    cfg.update({'backbone': {'depth': 3}})
    if 'cached' in cfg.dataset:
        # Do not use cache for testing because of memory issues
        cfg.dataset.cached = ''

    dataset_train, dataset_valid = get_dataset(cfg, cwd=cwd)
    assert isinstance(dataset_train, expected_dataclass)
    if expected_dataclass_valid is not None:
        assert isinstance(dataset_valid, expected_dataclass_valid)
    else:
        assert dataset_valid is None


@pytest.mark.parametrize('n_repeats', (1, 5, 10))
def test_dataset_repeat(n_repeats: int):
    dataset = Set12SyntheticDataset(path="../data/Set12", n_repeats=n_repeats, standardize=True, noise_param=0)
    assert len(dataset) == 12 * n_repeats
    assert torch.equal(dataset[(len(dataset) - 1) % 12]['image'], dataset[len(dataset) - 1]['image'])


@pytest.mark.parametrize('dataset_name', ['synthetic', 'synthetic_grayscale'])
def test_concat_dataset(dataset_name):
    # TODO add more meaningful tests
    cwd = Path('..')
    cfg = OmegaConf.load(f'../config/experiment/{dataset_name}.yaml')
    for i, d in enumerate(cfg.dataset_test.datasets):
        cfg.dataset_test.datasets[i].path = str(cwd / d.path)
        cfg.dataset_test.datasets[i] = OmegaConf.merge(cfg.dataset, cfg.dataset_test.datasets[i])
    _ = instantiate(cfg.dataset_test)
