import os
from pathlib import Path
from typing import Optional, Union, Tuple, Callable

import numpy as np
import pytest
import torch
from albumentations import PadIfNeeded
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from noise2same.dataset import *
from noise2same.dataset.getter import expand_dataset_cfg
from noise2same.dataset.util import mask_like_image, add_noise
from noise2same.util import crop_as_gt


@pytest.fixture(scope="module", autouse=True)
def set_cwd():
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}


@pytest.mark.parametrize("divisor", (2, 4, 8, 16, 32, 64, 127))
def test_crop_as(divisor: int):
    pad = PadIfNeeded(
        min_height=None,
        min_width=None,
        pad_height_divisor=divisor,
        pad_width_divisor=divisor,
    )

    image = np.random.uniform(size=(180, 180, 1))
    padded = pad(image=image)["image"]
    cropped = crop_as_gt(padded, image)
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


@pytest.mark.parametrize('zeros,alpha,sigma,sap',
                         [(np.zeros, 5, 0.1, 0.1),
                          (np.zeros, (1, 10), (0.1, 0.2), (0.1, 0.2)),
                          (torch.zeros, 5, 0.1, 0.1),
                          (torch.zeros, (1, 10), (0.1, 0.2), (0.1, 0.2))
                          ])
def test_noise_addition(
    zeros: Callable,
    alpha: Union[float, int, Tuple[Union[float, int]]],
    sigma: Union[float, int, Tuple[Union[float, int]]],
    sap: Union[float, int, Tuple[Union[float, int]]]
):
    shape = (1, 16, 64, 64)
    image = zeros(shape)
    noisy = add_noise(image, alpha=5, sigma=0.1, sap=0.1)
    assert image.shape == noisy.shape
    assert type(image) == type(noisy)
    assert image.dtype == noisy.dtype


@pytest.mark.parametrize('dataset_name,expected_dataclass,expected_dataclass_valid,expected_dataclass_test',
                         [('bsd68', BSD68Dataset, BSD68Dataset, BSD68Dataset),
                          # ('hanzi', HanziDataset, HanziDataset, HanziDataset), # TODO fix memory issue
                          # ('imagenet', ImagenetDataset, ImagenetDataset, ImagenetTestDataset), # TODO fix memory issue
                          ('microtubules', MicrotubulesDataset, None, MicrotubulesDataset),
                          ('microtubules_generated', MicrotubulesDataset, None, MicrotubulesTestDataset),
                          ('fmd', FMDDataset, FMDDataset, FMDDataset),
                          ('fmd_deconvolution', FMDDataset, FMDDataset, FMDDataset),
                          # ('planaria', PlanariaDataset, PlanariaDataset, PlanariaTestDataset), # TODO fix memory issue
                          # ('sidd', SIDDDataset, SIDDDataset, SIDDDataset), # TODO move dataset
                          ('synthetic', ImagenetSyntheticDataset, Set14SyntheticDataset, SyntheticTestDataset),
                          ('synthetic_grayscale', BSD400SyntheticDataset, BSD68SyntheticDataset, SyntheticTestDataset),
                          ('ssi', SSIDataset, None, SSIDataset),
                          ('hela_shallow', HelaShallowDataset, HelaShallowDataset, HelaShallowDataset),
                          ('hela', HelaDataset, HelaDataset, HelaDataset),
                          ])
def test_dataset_instantiation(
        dataset_name: str,
        expected_dataclass: type,
        expected_dataclass_valid: Optional[type],
        expected_dataclass_test: Optional[type]
):
    with initialize(version_base=None, config_path="../config/experiment", job_name="test"):
        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=["+cwd=${hydra.runtime.cwd}"])
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)

        if 'cached' in cfg.dataset_train:
            # Do not use cache for testing because of memory issues
            cfg.dataset_train.cached = ''

        print('\n', OmegaConf.to_yaml(cfg))

        if 'dataset_valid' in cfg:
            dataset_valid = instantiate(cfg.dataset_valid)
            assert isinstance(dataset_valid, expected_dataclass_valid)
            assert dataset_valid[0] is not None

        if 'dataset_test' in cfg:
            dataset_test = instantiate(cfg.dataset_test)
            assert isinstance(dataset_test, expected_dataclass_test)
            assert dataset_test[0] is not None

        dataset = instantiate(cfg.dataset_train)
        assert isinstance(dataset, expected_dataclass)
        assert str(dataset) is not None and str(dataset) != 'dataset'
        assert dataset[0] is not None


@pytest.mark.parametrize('n_repeats', (1, 5, 10))
def test_dataset_repeat(n_repeats: int):
    dataset = Set12SyntheticDataset(path="data/Set12", n_repeats=n_repeats, standardize=True, noise_param=0)
    assert len(dataset) == 12 * n_repeats
    assert torch.equal(dataset[(len(dataset) - 1) % 12]['image'], dataset[len(dataset) - 1]['image'])


@pytest.mark.parametrize('dataset_name', ['synthetic', 'synthetic_grayscale'])
def test_concat_dataset(dataset_name):
    # TODO add more meaningful tests
    with initialize(version_base=None, config_path="../config/experiment"):
        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=['+cwd=${hydra.runtime.cwd}'])
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))
        _ = instantiate(cfg.dataset_test)


