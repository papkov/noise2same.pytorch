from typing import Tuple, Dict, List

import numpy as np
import tifffile
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from noise2same.dataset.abc import AbstractNoiseDataset3DLarge
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


def get_test_dataset_and_gt(cfg: DictConfig) -> Tuple[Dataset, List[np.ndarray]]:
    """
    Collect test dataset and ground truth specified in the configuration
    :param cfg: DictConfig, training/evaluation configuration object
    :return: Tuple[Dataset, np.ndarray]
    """

    dataset_test = instantiate(cfg.dataset_test)
    # TODO this is a hack, should be fixed in the dataset
    if isinstance(dataset_test, AbstractNoiseDataset3DLarge):
        ground_truth = dataset_test.ground_truth
    else:
        # TODO move ground truth access into evaluation
        ground_truth = [elem['ground_truth'] for elem in dataset_test]
    # if not isinstance(dataset_test.ground_truth[0], np.ndarray):
    #     dataset_test.ground_truth = [dataset_test._get_image(image) for image in dataset_test.ground_truth]
    return dataset_test, ground_truth


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
