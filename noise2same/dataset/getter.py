from typing import Tuple, Optional, Dict

import numpy as np
import tifffile
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from noise2same.util import normalize_percentile
from . import planaria


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
    # TODO consider replacing with interpolation and eval https://github.com/omry/omegaconf/issues/91
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


def get_test_dataset_and_gt(cfg: DictConfig) -> Tuple[Dataset, np.ndarray]:
    """
    Collect test dataset and ground truth specified in the configuration
    :param cfg: DictConfig, training/evaluation configuration object
    :return: Tuple[Dataset, np.ndarray]
    """

    pad_divisor = compute_pad_divisor(cfg)
    dataset_test = instantiate(cfg.dataset_test, pad_divisor=pad_divisor)
    return dataset_test, dataset_test.ground_truth


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
