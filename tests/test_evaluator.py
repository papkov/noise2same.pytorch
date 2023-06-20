import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from noise2same.dataset.getter import expand_dataset_cfg
from noise2same.denoiser import Denoiser
from noise2same.evaluator import Evaluator


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
    dataset.image_index = {k: v[:2] for k, v in dataset.image_index.items()}
    evaluator = Evaluator(Denoiser(), device='cpu')
    _ = evaluator.evaluate(dataset, factory)
