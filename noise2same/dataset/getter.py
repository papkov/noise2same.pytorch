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


def compute_pad_divisor(cfg: DictConfig):
    if cfg.backbone_name == 'unet':
        return 2 ** cfg.backbone.depth
    elif cfg.backbone_name == 'swinir':
        return cfg.backbone.patch_size * cfg.backbone.window_size
    else:
        return 1


def get_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset]:

    cwd = Path(get_original_cwd())
    dataset_valid = None

    pad_divisor = compute_pad_divisor(cfg)

    if cfg.experiment.lower() == "bsd68":
        dataset_train = bsd68.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/",
            mode="train",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            pad_divisor=pad_divisor,
        )
        if cfg.training.validate:
            dataset_valid = bsd68.BSD68DatasetPrepared(
                path=cwd / "data/BSD68/", mode="val",
                pad_divisor=pad_divisor,
            )

    elif cfg.experiment.lower() == "hanzi":
        dataset_train = hanzi.HanziDatasetPrepared(
            path=cwd / "data/Hanzi/tiles",
            mode="training",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            version=cfg.data.version,
            noise_level=cfg.data.noise_level,
            pad_divisor=pad_divisor,
        )
        if cfg.training.validate:
            dataset_valid = hanzi.HanziDatasetPrepared(
                path=cwd / "data/Hanzi/tiles",
                mode="validation",
                version=cfg.data.version,
                noise_level=cfg.data.noise_level,
                pad_divisor=pad_divisor,
            )

    elif cfg.experiment.lower() == "imagenet":
        dataset_train = imagenet.ImagenetDatasetPrepared(
            path=cwd / "data/ImageNet",
            mode="train",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            version=cfg.data.version,
            pad_divisor=pad_divisor,
        )
        if cfg.training.validate:
            dataset_valid = imagenet.ImagenetDatasetPrepared(
                path=cwd / "data/ImageNet",
                mode="val",
                version=cfg.data.version,
                pad_divisor=pad_divisor,
            )

    elif cfg.experiment.lower() == "planaria":
        dataset_train = planaria.PlanariaDatasetPrepared(
            path=cwd / "data/Denoising_Planaria",
            mode="train",
            transforms=training_augmentations_3d(),
            pad_divisor=pad_divisor,
        )
        if cfg.training.validate:
            dataset_valid = planaria.PlanariaDatasetPrepared(
                path=cwd / "data/Denoising_Planaria",
                mode="val",
                pad_divisor=pad_divisor,
            )

    elif cfg.experiment.lower() == "microtubules":
        dataset_train = microtubules.MicrotubulesDataset(
            path=cwd / cfg.data.path,
            input_name=cfg.data.input_name,
            transforms=training_augmentations_3d(),
            tile_size=cfg.data.tile_size,
            tile_step=cfg.data.tile_step,
            add_blur_and_noise=cfg.data.add_blur_and_noise,
            pad_divisor=pad_divisor,
        )

    elif cfg.experiment.lower() == "ssi":
        dataset_train = ssi.SSIDataset(
            path=cwd / cfg.data.path,
            input_name=cfg.data.input_name,
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            pad_divisor=pad_divisor,
        )
    else:
        # todo add other datasets
        raise ValueError

    return dataset_train, dataset_valid


def get_test_dataset_and_gt(cfg: DictConfig) -> Tuple[Dataset, np.ndarray]:

    pad_divisor = compute_pad_divisor(cfg)

    cwd = Path(get_original_cwd())
    if cfg.experiment.lower() == "bsd68":
        dataset = bsd68.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/",
            mode="test",
            pad_divisor=pad_divisor,
        )
        gt = np.load(
            str(cwd / "data/BSD68/test/bsd68_groundtruth.npy"), allow_pickle=True
        )

    elif cfg.experiment.lower() == "hanzi":
        dataset = hanzi.HanziDatasetPrepared(
            path=cwd / "data/Hanzi/tiles",
            mode="testing",
            pad_divisor=pad_divisor,
        )
        gt = np.load(str(cwd / "data/Hanzi/tiles/testing.npy"))[:, 0]

    elif cfg.experiment.lower() == "imagenet":
        dataset = imagenet.ImagenetDatasetTest(
            path=cwd / "data/ImageNet/",
            pad_divisor=pad_divisor,
        )
        gt = [
            np.load(p)[0] for p in tqdm(sorted((dataset.path / "test").glob("*.npy")))
        ]

    elif cfg.experiment.lower() == "planaria":
        # This returns just a single image!
        # Use get_planaria_dataset_and_gt() instead
        dataset = planaria.PlanariaDatasetTiff(
            cwd
            / "data/Denoising_Planaria/test_data/condition_1/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0012.tif",
            standardize=True,
            pad_divisor=pad_divisor,
        )
        dataset.mean, dataset.std = 0, 1

        gt = tifffile.imread(
            cwd
            / "data/Denoising_Planaria/test_data/GT/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0012.tif"
        )
        gt = normalize_percentile(gt, 0.1, 99.9)

    elif cfg.experiment.lower() == "microtubules":
        dataset = microtubules.MicrotubulesDataset(
            path=cwd / cfg.data.path,
            input_name=cfg.data.input_name,
            # we can double the size of the tiles for validation
            tile_size=cfg.data.tile_size * 2,  # 64 * 2 = 128
            tile_step=cfg.data.tile_step * 2,  # 48 * 2 = 96
            add_blur_and_noise=cfg.data.add_blur_and_noise,  # TODO add different noise by random seed?
            pad_divisor=pad_divisor,
        )
        # dataset.mean, dataset.std = 0, 1

        gt = io.imread(str(cwd / "data/microtubules-simulation/ground-truth.tif"))
        gt = normalize_percentile(gt, 0.1, 99.9)

    elif cfg.experiment.lower() == "ssi":
        dataset = ssi.SSIDataset(
            path=cwd / cfg.data.path,
            input_name=cfg.data.input_name,
            pad_divisor=pad_divisor,
        )
        gt = dataset.gt
    else:
        raise ValueError(f"Dataset {cfg.experiment} not found")

    return dataset, gt


def get_planaria_dataset_and_gt(filename_gt: str):
    gt = tifffile.imread(filename_gt)
    gt = normalize_percentile(gt, 0.1, 99.9)
    datasets = {}
    for c in range(1, 4):
        datasets[f"c{c}"] = planaria.PlanariaDatasetTiff(
            filename_gt.replace("GT", f"condition_{c}"),
            standardize=True,
        )
        datasets[f"c{c}"].mean, datasets[f"c{c}"].std = 0, 1

    return datasets, gt
