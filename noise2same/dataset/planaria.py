from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tifffile
from pytorch_toolbelt.inference.tiles import ImageSlicer

from noise2same.dataset.abc import AbstractNoiseDataset3D, AbstractNoiseDataset3DLarge
from noise2same.util import normalize_percentile


@dataclass
class PlanariaDataset(AbstractNoiseDataset3D):
    path: Union[Path, str] = "data/Denoising_Planaria"
    mode: str = "train"
    train_size: float = 0.9
    standardize: bool = False  # data was prepared and percentile normalized

    def _get_images(self) -> Union[List[str], np.ndarray]:
        data = np.load(self.path / "train_data/data_label.npz")["X"].astype(np.float32)
        if self.mode == "train":
            data = data[: int(len(data) * self.train_size)]
        else:
            data = data[int(len(data) * self.train_size) :]
        return data

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path


@dataclass
class PlanariaTiffDataset(AbstractNoiseDataset3DLarge):
    tile_size: int = 256
    tile_step: int = 192
    crop_border: int = 32
    weight: str = "pyramid"

    def _get_images(self) -> Union[List[str], np.ndarray]:
        self.image = tifffile.imread(self.path)[..., None]

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
        return self.tiler.crops

    def _read_image(self, image_or_path: List[int]) -> Tuple[np.ndarray, List[int]]:
        image, crop = self.tiler.crop_tile(image=self.image, crop=image_or_path)
        return np.moveaxis(image, -1, 0), crop
