from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import tifffile
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from noise2same.util import normalize_percentile
from . import bsd68, fmd, hanzi, imagenet, sidd, microtubules, planaria, ssi, synthetic, synthetic_grayscale


def expand_dataset_cfg(cfg: DictConfig) -> None:
    """
    Expands dataset_valid and dataset_test configs with parameters from dataset config
    :param cfg: config to modify inplace, should contain dataset, optionally dataset_valid, dataset_test
    :return: None
    """
    OmegaConf.set_struct(cfg.dataset, False)  # necessary to create new keys
    for dataset_key in ('dataset_valid', 'dataset_test'):
        if dataset_key in cfg:
            if 'datasets' in cfg[dataset_key]:
                # If the dataset is composed of multiple datasets, update each of them
                for i, dataset in enumerate(cfg[dataset_key].datasets):
                    cfg[dataset_key].datasets[i] = OmegaConf.merge(cfg.dataset, dataset)
            else:
                # Validation dataset config updates fields of the training dataset config
                cfg[dataset_key] = OmegaConf.merge(cfg.dataset, cfg[dataset_key])


def compute_pad_divisor(cfg: DictConfig) -> Optional[int]:
    """
    Compute the number by which the padded image size should
    be divisible, so that it is suitable for the chosen backbone
    :param cfg: DictConfig, training/evaluation configuration object
    :return: Optional[int]
    """
    if cfg.backbone_name == "unet":
        return 2 ** cfg.backbone.depth
    elif cfg.backbone_name in ("swinir", "bsp_swinir"):
        return cfg.backbone.window_size
    elif cfg.backbone_name == "swinia":
        return cfg.backbone.window_size * max(cfg.backbone.dilations) * max(cfg.backbone.shuffles)
    else:
        raise ValueError("Incorrect backbone name")


def get_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset]:
    """
    Collect training and validation datasets specified in the configuration
    :param cfg: DictConfig, training/evaluation configuration object
    :return: Tuple[Dataset, Dataset]
    """
    # TODO consider moving to main
    dataset_valid = None
    pad_divisor = compute_pad_divisor(cfg)

    if 'dataset_valid' in cfg:
        dataset_valid = instantiate(cfg.dataset_valid, pad_divisor=pad_divisor)

    dataset_train = instantiate(cfg.dataset, pad_divisor=pad_divisor)
    return dataset_train, dataset_valid


def get_test_dataset_and_gt(cfg: DictConfig, cwd: Path) -> Tuple[Dataset, np.ndarray]:
    """
    Collect test dataset and ground truth specified in the configuration
    :param cfg: DictConfig, training/evaluation configuration object
    :param cwd: Path, project working directory
    :return: Tuple[Dataset, np.ndarray]
    """

    pad_divisor = compute_pad_divisor(cfg)

    if cfg.experiment.lower() == "bsd68":
        dataset = bsd68.BSD68Dataset(
            path=cwd / "data/BSD68/",
            mode="test",
            pad_divisor=pad_divisor,
        )

    elif cfg.experiment.lower() == "fmd":
        dataset = fmd.FMDDataset(
            path=cwd / "data/FMD",
            mode="test",
            pad_divisor=pad_divisor,
            part=cfg.dataset.part,
            add_blur_and_noise=cfg.dataset.add_blur_and_noise,
        )

    elif cfg.experiment.lower() == "synthetic":
        params = {
            "noise_type": cfg.dataset.noise_type,
            "noise_param": cfg.dataset.noise_param,
            "pad_divisor": pad_divisor,
            "standardize": cfg.dataset.standardize,
        }
        dataset = synthetic.SyntheticTestDataset(
            [partial(synthetic.KodakSyntheticDataset, path=cwd / "data/Kodak"),
             partial(synthetic.BSD300SyntheticDataset, path=cwd / "data/BSD300/test"),
             partial(synthetic.Set14SyntheticDataset, path=cwd / "data/Set14"),
             ],
            **params,
        )

    elif cfg.experiment.lower() == "synthetic_grayscale":
        params = {
            "noise_type": cfg.dataset.noise_type,
            "noise_param": cfg.dataset.noise_param,
            "pad_divisor": pad_divisor,
            "standardize": cfg.dataset.standardize,
        }
        dataset = synthetic.SyntheticTestDataset(
            [partial(synthetic_grayscale.Set12SyntheticDataset, path=cwd / "data/Set12"),
             partial(synthetic_grayscale.BSD68SyntheticDataset, path=cwd / "data/BSD68-test", fixed=False),
             ],
            **params,
        )

    elif cfg.experiment.lower() == "hanzi":
        dataset = hanzi.HanziDataset(
            path=cwd / "data/Hanzi/tiles",
            mode="testing",
            pad_divisor=pad_divisor,
            noise_level=cfg.dataset.noise_level
        )

    elif cfg.experiment.lower() == "imagenet":
        dataset = imagenet.ImagenetTestDataset(
            path=cwd / "data/ImageNet/",
            pad_divisor=pad_divisor,
        )

    elif cfg.experiment.lower() == "sidd":
        dataset = sidd.SIDDDataset(
            path=cwd / "data/SIDD-NAFNet/",
            mode='test',
            pad_divisor=pad_divisor,
        )

    elif cfg.experiment.lower() == "planaria":
        # This returns just a single image!
        # Use get_planaria_dataset_and_gt() instead
        dataset = planaria.PlanariaTiffDataset(
            cwd
            / "data/Denoising_Planaria/test_data/condition_1/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0012.tif",
            standardize=True,
            pad_divisor=pad_divisor,
        )
        dataset.mean, dataset.std = 0, 1

    elif cfg.experiment.lower() == "microtubules":
        dataset = microtubules.MicrotubulesDataset(
            path=cwd / cfg.dataset.path,
            input_name=cfg.dataset.input_name,
            # we can double the size of the tiles for validation
            tile_size=cfg.dataset.tile_size * 2,  # 64 * 2 = 128
            tile_step=cfg.dataset.tile_step * 2,  # 48 * 2 = 96
            add_blur_and_noise=cfg.dataset.add_blur_and_noise,  # TODO add different noise by random seed?
            pad_divisor=pad_divisor,
        )
        # dataset.mean, dataset.std = 0, 1

    elif cfg.experiment.lower() == "ssi":
        dataset = ssi.SSIDataset(
            path=cwd / cfg.dataset.path,
            input_name=cfg.dataset.input_name,
            pad_divisor=pad_divisor,
        )
    else:
        raise ValueError(f"Dataset {cfg.experiment} not found")

    return dataset, dataset.ground_truth


def get_planaria_dataset_and_gt(filename_gt: str) -> Tuple[Dict[str, Dataset], np.ndarray]:
    """
    Collect Planaria dataset and ground truth
    :param filename_gt: str, Planaria dataset ground truth filename
    :return: Tuple[Dict[str, Dataset], np.ndarray]
    """
    gt = tifffile.imread(filename_gt)
    gt = normalize_percentile(gt, 0.1, 99.9)
    datasets = {}
    for c in range(1, 4):
        datasets[f"c{c}"] = planaria.PlanariaTiffDataset(
            filename_gt.replace("GT", f"condition_{c}"),
            standardize=True,
        )
        datasets[f"c{c}"].mean, datasets[f"c{c}"].std = 0, 1

    return datasets, gt
