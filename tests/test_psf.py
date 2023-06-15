import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from noise2same.dataset.getter import get_dataset, expand_dataset_cfg
from noise2same.psf.psf_convolution import PSF, PSFParameter


def test_psf_fft(s: int = 7):
    kernel = np.random.rand(s, s, s)
    patch = torch.rand(1, 1, 64, 64, 64)

    psf = PSF(kernel, fft=False)
    psf_fft = PSF(kernel, fft=True)

    psf_out = psf(patch)
    psf_fft_out = psf_fft(patch)

    assert patch.shape == psf_out.shape
    assert torch.allclose(psf_out, psf_fft_out)


def test_psf_delta(s: int = 7):
    kernel = np.zeros((s, s, s))
    kernel[s // 2, s // 2, s // 2] = 1  # delta function

    patch = torch.rand(1, 1, 64, 64, 64)

    psf = PSF(kernel, fft=False)
    psf_out = psf(patch)

    assert torch.allclose(psf_out, patch)


def test_psf_fft_delta(s: int = 7):
    kernel = np.zeros((s, s, s))
    kernel[s // 2, s // 2, s // 2] = 1  # delta function

    patch = torch.rand(1, 1, 64, 64, 64)

    psf = PSF(kernel, fft=True)
    psf_out = psf(patch)

    assert torch.allclose(psf_out, patch, atol=1e-6)


def test_psf_parameter():
    kernel = np.random.rand(7, 7, 7)
    patch = torch.rand(1, 1, 64, 64, 64)

    psf = PSF(kernel, fft=True)
    psf_param = PSFParameter(kernel, fft=True)

    psf_out = psf(patch)
    psf_param_out = psf_param(patch)

    assert patch.shape == psf_out.shape
    assert torch.allclose(psf_out, psf_param_out)


def test_large_psf():
    kernel = np.random.rand(128, 256, 512)
    patch = torch.rand(1, 1, 128, 128, 128)

    psf = PSFParameter(kernel, fft=True)
    psf_out = psf(patch)

    assert patch.shape == psf_out.shape


def test_psf_auto_padding():
    kernel = np.random.rand(7, 7, 7)
    patch = torch.rand(1, 1, 64, 64, 64)

    psf = PSFParameter(kernel, fft=True, auto_padding=False)
    psf_auto = PSFParameter(kernel, fft=True, auto_padding=True)

    psf_out = psf(patch)
    psf_auto_out = psf_auto(patch)

    assert patch.shape == psf_auto_out.shape
    assert torch.allclose(psf_out, psf_auto_out)


@pytest.mark.parametrize("dataset_name", ["fmd_deconvolution",
                                          "microtubules",
                                          "microtubules_generated",
                                          "microtubules_original",
                                          "ssi",
                                          ])
def test_psf_instantiation(dataset_name):
    os.chdir(Path(__file__).parent.parent)  # necessary to resolve interpolations as ${hydra.runtime.cwd}
    with initialize(version_base=None, config_path="../config/experiment"):
        overrides = ['+backbone_name=unet', '+backbone.depth=3', '+dataset.n_channels=1']
        cfg = compose(config_name=dataset_name, return_hydra_config=True, overrides=overrides)
        OmegaConf.resolve(cfg)  # resolves interpolations as ${hydra.runtime.cwd}
        expand_dataset_cfg(cfg)
        print('\n', OmegaConf.to_yaml(cfg))

    dataset_train, dataset_valid = get_dataset(cfg)

    kernel_psf = getattr(dataset_train, "psf", None)
    if kernel_psf is not None:
        psf = instantiate(cfg.psf, kernel_psf=kernel_psf)
    else:
        psf = instantiate(cfg.psf)

    assert isinstance(psf, PSFParameter)
