from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from skimage import io
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from noise2same.dataset import bsd68, hanzi, imagenet, microtubules, planaria, ssi
from noise2same.dataset.util import training_augmentations_2d, training_augmentations_3d
from noise2same.util import normalize_percentile


def get_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset]:

    cwd = Path(get_original_cwd())
    dataset_valid = None

    if cfg.name.lower() == "bsd68":
        dataset_train = bsd68.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/",
            mode="train",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
        )
        if cfg.training.validate:
            dataset_valid = bsd68.BSD68DatasetPrepared(
                path=cwd / "data/BSD68/", mode="val"
            )

    elif cfg.name.lower() == "hanzi":
        dataset_train = hanzi.HanziDatasetPrepared(
            path=cwd / "data/Hanzi/tiles",
            mode="training",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            version=cfg.data.version,
            noise_level=cfg.data.noise_level,
        )
        if cfg.training.validate:
            dataset_valid = hanzi.HanziDatasetPrepared(
                path=cwd / "data/Hanzi/tiles",
                mode="validation",
                version=cfg.data.version,
                noise_level=cfg.data.noise_level,
            )

    elif cfg.name.lower() == "imagenet":
        dataset_train = imagenet.ImagenetDatasetPrepared(
            path=cwd / "data/ImageNet",
            mode="train",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            version=cfg.data.version,
        )
        if cfg.training.validate:
            dataset_valid = imagenet.ImagenetDatasetPrepared(
                path=cwd / "data/ImageNet",
                mode="val",
                version=cfg.data.version,
            )

    elif cfg.name.lower() == "planaria":
        dataset_train = planaria.PlanariaDatasetPrepared(
            path=cwd / "data/Denoising_Planaria",
            mode="train",
            transforms=training_augmentations_3d(),
        )
        if cfg.training.validate:
            dataset_valid = planaria.PlanariaDatasetPrepared(
                path=cwd / "data/Denoising_Planaria",
                mode="val",
            )

    elif cfg.name.lower() == "microtubules":
        dataset_train = microtubules.MicrotubulesDataset(
            path=cwd / cfg.data.path,
            input_name=cfg.data.input_name,
            transforms=training_augmentations_3d(),
            tile_size=cfg.data.tile_size,
            tile_step=cfg.data.tile_step,
        )

    elif cfg.name.lower() == "ssi":
        dataset_train = ssi.SSIDataset(
            path=cwd / cfg.data.path,
            input_name=cfg.data.input_name,
            transforms=training_augmentations_2d(crop=cfg.training.crop),
        )
    else:
        # todo add other datasets
        raise ValueError

    return dataset_train, dataset_valid


def get_test_dataset_and_gt(cfg: DictConfig) -> Tuple[Dataset, np.ndarray]:

    cwd = Path(get_original_cwd())
    if cfg.name.lower() == "bsd68":
        dataset = bsd68.BSD68DatasetPrepared(path=cwd / "data/BSD68/", mode="test")
        gt = np.load(
            str(cwd / "data/BSD68/test/bsd68_groundtruth.npy"), allow_pickle=True
        )

    elif cfg.name.lower() == "hanzi":
        dataset = hanzi.HanziDatasetPrepared(
            path=cwd / "data/Hanzi/tiles", mode="testing"
        )
        gt = np.load(str(cwd / "data/Hanzi/tiles/testing.npy"))[:, 0]

    elif cfg.name.lower() == "imagenet":
        dataset = imagenet.ImagenetDatasetTest(path=cwd / "data/ImageNet/")
        gt = [
            np.load(p)[0] for p in tqdm(sorted((dataset.path / "test").glob("*.npy")))
        ]

    elif cfg.name.lower() == "planaria":
        # This returns just a single image!
        # Use get_planaria_dataset_and_gt() instead
        dataset = planaria.PlanariaDatasetTiff(
            cwd
            / "data/Denoising_Planaria/test_data/condition_1/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0012.tif",
            standardize=True,
        )
        dataset.mean, dataset.std = 0, 1

        gt = tifffile.imread(
            cwd
            / "data/Denoising_Planaria/test_data/GT/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0012.tif"
        )
        gt = normalize_percentile(gt, 0.1, 99.9)

    elif cfg.name.lower() == "microtubules":
        dataset = microtubules.MicrotubulesDataset(
            path=cfg.data.path,
            input_name=cfg.data.input_name,
            tile_size=cfg.data.crop,
            tile_step=cfg.data.crop - cfg.data.crop // 4,
        )
        # dataset.mean, dataset.std = 0, 1

        gt = io.imread(str(cwd / "data/microtubules-simulation/ground-truth.tif"))
        gt = normalize_percentile(gt, 0.1, 99.9)
    else:
        raise ValueError(f"Dataset {cfg.name} not found")

    return dataset, gt


def get_planaria_dataset_and_gt(filename_gt: str):
    gt = tifffile.imread(filename_gt)
    gt = normalize_percentile(gt, 0.1, 99.9)
    datasets = {}
    for c in range(1, 4):
        datasets[f"c{c}"] = planaria.PlanariaDatasetTiff(
            filename_gt.replace("GT", f"condition_{c}"), standardize=True
        )
        datasets[f"c{c}"].mean, datasets[f"c{c}"].std = 0, 1

    return datasets, gt
