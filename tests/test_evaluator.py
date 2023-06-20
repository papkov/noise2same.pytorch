import os
from pathlib import Path
import numpy as np
import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from noise2same.dataset.getter import expand_dataset_cfg
from noise2same.denoiser import Denoiser
from noise2same.evaluator import Evaluator
from torch.utils.data import Subset


class SubsetAttr(Subset):
    """
    Wrapper for Subset that allows to access attributes of the wrapped dataset.
    """

    def __getattr__(self, item):
        return getattr(self.dataset, item)


@pytest.fixture(scope="module", autouse=True)
def set_cwd():
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}


@pytest.mark.parametrize('dataset_name',
                         ['bsd68', 'hanzi', 'imagenet', 'fmd', 'fmd_deconvolution', 'synthetic',
                          'synthetic_grayscale', 'ssi', 'hela_shallow'])
def test_regular_dataset_inference(dataset_name: str):
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+cwd=${hydra.runtime.cwd}']
        if dataset_name == 'synthetic':
            # Do not use cache for testing because of memory issues
            overrides.append('dataset.cached=null')

        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset_test)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    evaluator = Evaluator(Denoiser(), device='cpu')
    batch = next(iter(loader))
    out, _ = evaluator._inference_batch(batch)
    predictions = evaluator._revert_batch(out, ['image'])
    original = evaluator._revert_batch(batch, ['image'])
    for shape, pred, orig in zip(batch['shape'], predictions['image'], original['image']):
        assert np.all(np.array(shape) == np.array(pred.shape))
        assert np.allclose(pred, orig)


@pytest.mark.parametrize('dataset_name', ['imagenet', 'planaria'])
def test_tiled_dataset_inference(dataset_name: str):
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+cwd=${hydra.runtime.cwd}']
        if dataset_name == 'synthetic':
            # Do not use cache for testing because of memory issues
            overrides.append('dataset.cached=null')

        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset_test)
    factory = instantiate(cfg.factory_test)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    evaluator = Evaluator(Denoiser(), device='cpu')
    batch = next(iter(loader))
    out, _ = evaluator._inference_large_batch(batch, factory)
    predictions = evaluator._revert_batch(out, ['image'])
    original = evaluator._revert_batch(batch, ['image'])
    for shape, pred, orig in zip(batch['shape'], predictions['image'], original['image']):
        assert np.all(np.array(shape) == np.array(pred.shape))
        assert np.allclose(pred, orig)


@pytest.mark.parametrize('dataset_name', ['imagenet', 'planaria'])
def test_tiled_dataset_evaluation(dataset_name: str):
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+cwd=${hydra.runtime.cwd}']
        if dataset_name == 'synthetic':
            # Do not use cache for testing because of memory issues
            overrides.append('dataset.cached=null')

        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset_test)
    dataset = SubsetAttr(dataset, range(min(2, len(dataset))))
    factory = instantiate(cfg.factory_test) if 'factory_test' in cfg else None
    evaluator = Evaluator(Denoiser(), device='cpu')
    outputs = evaluator.evaluate(dataset, factory, metrics=('rmse',))
    assert outputs is not None
