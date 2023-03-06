from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import tifffile
from omegaconf import DictConfig
from skimage import io
from torch.utils.data import Dataset, ConcatDataset
from tqdm.auto import tqdm

from . import bsd68, fmd, hanzi, imagenet, sidd, microtubules, planaria, ssi, synthetic, synthetic_grayscale
from .util import training_augmentations_2d, training_augmentations_3d, validation_transforms_2d
from noise2same.util import normalize_percentile


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


def get_dataset(cfg: DictConfig, cwd: Path) -> Tuple[Dataset, Dataset]:
    """
    Collect training and validation datasets specified in the configuration
    :param cfg: DictConfig, training/evaluation configuration object
    :param cwd: Path, project working directory
    :return: Tuple[Dataset, Dataset]
    """

    dataset_valid = None

    pad_divisor = compute_pad_divisor(cfg)

    transforms = None
    transforms_valid = None
    if cfg.experiment.lower() in ("bsd68", "fmd", "synthetic", "synthetic_grayscale", "hanzi", "imagenet", "sidd", "ssi"):
        transforms = training_augmentations_2d(crop=cfg.training.crop)
        transforms_valid = validation_transforms_2d(crop=cfg.training.crop)

    if cfg.experiment.lower() == "bsd68":
        dataset_train = bsd68.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/",
            mode="train",
            transforms=transforms,
            pad_divisor=pad_divisor,
        )
        if cfg.training.validate:
            dataset_valid = bsd68.BSD68DatasetPrepared(
                path=cwd / "data/BSD68/", mode="val",
                pad_divisor=pad_divisor,
            )

    elif cfg.experiment.lower() == "synthetic":
        dataset_train = synthetic.ImagenetSyntheticDataset(
            path=cwd / "data/Imagenet_val",
            noise_type=cfg.data.noise_type,
            noise_param=cfg.data.noise_param,
            transforms=transforms,
            pad_divisor=pad_divisor,
            standardize=cfg.data.standardize,
        )
        if cfg.training.validate:
            dataset_valid = synthetic.Set14SyntheticDataset(
                path=cwd / "data/Set14",
                noise_type=cfg.data.noise_type,
                noise_param=cfg.data.noise_param,
                transforms=transforms_valid,
                pad_divisor=pad_divisor,
                standardize=cfg.data.standardize,
            )

    elif cfg.experiment.lower() == "synthetic_grayscale":
        dataset_train = synthetic_grayscale.BSD400SyntheticDataset(
            path=cwd / "data/BSD400",
            noise_type=cfg.data.noise_type,
            noise_param=cfg.data.noise_param,
            transforms=transforms,
            pad_divisor=pad_divisor,
            standardize=cfg.data.standardize,
        )
        if cfg.training.validate:
            dataset_valid = synthetic_grayscale.BSD68SyntheticDataset(
                path=cwd / "data/BSD68-test",
                noise_type=cfg.data.noise_type,
                noise_param=cfg.data.noise_param,
                transforms=transforms_valid,
                pad_divisor=pad_divisor,
                standardize=cfg.data.standardize,
                fixed=True,
            )

    elif cfg.experiment.lower() == "fmd":
        dataset_train = fmd.FMDDatasetPrepared(
            path=cwd / "data/FMD",
            mode="train",
            transforms=transforms,
            pad_divisor=pad_divisor,
            part=cfg.data.part
        )
        if cfg.training.validate:
            dataset_valid = fmd.FMDDatasetPrepared(
                path=cwd / "data/FMD",
                mode="val",
                pad_divisor=pad_divisor,
                part=cfg.data.part
            )

    elif cfg.experiment.lower() == "hanzi":
        dataset_train = hanzi.HanziDatasetPrepared(
            path=cwd / "data/Hanzi/tiles",
            mode="training",
            transforms=transforms,
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
            transforms=transforms,
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

    elif cfg.experiment.lower() == "sidd":
        dataset_train = sidd.SIDDDatasetPrepared(
            path=cwd / "data/SIDD-NAFNet",
            mode="train",
            transforms=transforms,
            pad_divisor=pad_divisor,
        )
        if cfg.training.validate:
            dataset_valid = sidd.SIDDDatasetPrepared(
                path=cwd / "data/SIDD-NAFNet",
                mode="val",
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
            transforms=transforms,
            pad_divisor=pad_divisor,
        )
    else:
        # todo add other datasets
        raise ValueError(f"Unknown experiment: {cfg.experiment}")

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
        dataset = bsd68.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/",
            mode="test",
            pad_divisor=pad_divisor,
        )
        gt = np.load(
            str(cwd / "data/BSD68/test/bsd68_groundtruth.npy"), allow_pickle=True
        )

    elif cfg.experiment.lower() == "fmd":
        dataset = fmd.FMDDatasetPrepared(
            path=cwd / "data/FMD",
            mode="test",
            pad_divisor=pad_divisor,
            part=cfg.data.part
        )
        gt = dataset.ground_truth

    elif cfg.experiment.lower() == "synthetic":
        params = {
            "noise_type": cfg.data.noise_type,
            "noise_param": cfg.data.noise_param,
            "pad_divisor": pad_divisor,
            "standardize": cfg.data.standardize,
        }
        dataset = {
            "kodak": synthetic.KodakSyntheticDataset(path=cwd / "data/Kodak", **params),
            "bsd300": synthetic.BSD300SyntheticDataset(path=cwd / "data/BSD300/test", **params),
            "set14": synthetic.Set14SyntheticDataset(path=cwd / "data/Set14", **params),
        }
        gt = {name: [synthetic.read_image(p) for p in tqdm(ds.images, desc=name)] for name, ds in dataset.items()}

        # Repeat datasets for stable validation
        # https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/2fff2978/train.py#L412
        repeats = {"kodak": 10, "bsd300": 3, "set14": 20}
        dataset = {name: ConcatDataset([ds] * repeats[name]) for name, ds in dataset.items()}
        gt = {name: ds * repeats[name] for name, ds in gt.items()}

        # Concatenate datasets together
        dataset = ConcatDataset(list(dataset.values()))
        gt = np.concatenate(list(gt.values()))
        assert len(dataset) == len(gt)

    elif cfg.experiment.lower() == "synthetic_grayscale":
        params = {
            "noise_type": cfg.data.noise_type,
            "noise_param": cfg.data.noise_param,
            "pad_divisor": pad_divisor,
            "standardize": cfg.data.standardize,
        }
        dataset = {
            "set12": synthetic_grayscale.Set12SyntheticDataset(path=cwd / "data/Set12", **params),
            "bsd68": synthetic_grayscale.BSD68SyntheticDataset(path=cwd / "data/BSD68-test", fixed=True, **params),
        }
        gt = {name: [synthetic.read_image(p) for p in
                     tqdm(ds.ground_truth if ds.ground_truth is not None else ds.images, desc=name)]
              for name, ds in dataset.items()}

        # Repeat datasets for stable validation
        # https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/2fff2978/train.py#L412
        repeats = {"set12": 1, "bsd68": 1}
        dataset = {name: ConcatDataset([ds] * repeats[name]) for name, ds in dataset.items()}
        gt = {name: ds * repeats[name] for name, ds in gt.items()}

        # Concatenate datasets together
        dataset = ConcatDataset(list(dataset.values()))
        gt = np.concatenate(list(gt.values()))
        assert len(dataset) == len(gt)

    elif cfg.experiment.lower() == "hanzi":
        dataset = hanzi.HanziDatasetPrepared(
            path=cwd / "data/Hanzi/tiles",
            mode="testing",
            pad_divisor=pad_divisor,
            noise_level=cfg.data.noise_level
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

    elif cfg.experiment.lower() == "sidd":
        dataset = sidd.SIDDDatasetPrepared(
            path=cwd / "data/SIDD-NAFNet/",
            mode='test',
            pad_divisor=pad_divisor,
        )
        gt = dataset.ground_truth

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
        datasets[f"c{c}"] = planaria.PlanariaDatasetTiff(
            filename_gt.replace("GT", f"condition_{c}"),
            standardize=True,
        )
        datasets[f"c{c}"].mean, datasets[f"c{c}"].std = 0, 1

    return datasets, gt
