from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tifffile
from pytorch_toolbelt.inference.tiles import ImageSlicer

from noise2same.dataset.abc import AbstractNoiseDataset, AbstractNoiseDataset3DLarge
from noise2same.util import normalize_percentile
import re


@dataclass
class PlanariaDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/Denoising_Planaria"
    mode: str = "train"
    train_size: float = 0.9
    standardize: bool = False  # data was prepared and percentile normalized
    channel_last: bool = False
    n_dim: int = 3

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        data = np.load(self.path / "train_data/data_label.npz", mmap_mode='r')["X"].astype(np.float32)
        if self.mode == "train":
            data = data[: int(len(data) * self.train_size)]
        else:
            data = data[int(len(data) * self.train_size):]
        return {'noisy_input': data}

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {'image': self.image_index['noisy_input'][i]}


@dataclass
class PlanariaTiffDataset(AbstractNoiseDataset3DLarge):
    tile_size: int = 256
    tile_step: int = 192
    crop_border: int = 32
    weight: str = "pyramid"

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        self.image = tifffile.imread(self.path)[..., None]
        self.ground_truth = normalize_percentile(tifffile.imread(re.sub(r'condition_\d', 'GT', str(self.path))),
                                                 0.1, 99.9)

        if self.standardize:
            self.mean = self.image.mean()
            self.std = self.image.std()
            self.image = (self.image - self.mean) / self.std
        else:
            self.image = normalize_percentile(self.image)

        self.tiler = ImageSlicer(
            self.image.shape,
            tile_size=(96, self.tile_size, self.tile_size),
            tile_step=(96, self.tile_step, self.tile_step),
            weight=self.weight,
            is_channels=True,
            crop_border=(0, self.crop_border, self.crop_border),
        )
        return {'noisy_input': self.tiler.crops}

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image, crop = self.tiler.crop_tile(image=self.image, crop=self.image_index['noisy_input'][i])
        return {'image': np.moveaxis(image, -1, 0), 'ground_truth': self.ground_truth, 'crop': crop}
