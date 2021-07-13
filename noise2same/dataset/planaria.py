from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tifffile
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer

from noise2same.dataset.abc import AbstractNoiseDataset3D
from noise2same.util import normalize_percentile


@dataclass
class PlanariaDatasetPrepared(AbstractNoiseDataset3D):
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


class PlanariaDatasetTiff(AbstractNoiseDataset3D):
    path: Union[Path, str]
    tile_size: int = 256
    tile_step: int = 224
    weight: str = "pyramid"
    mean: float = 0
    std: float = 1
    standardize: bool = False

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
        )
        return self.tiler.crops

    def _read_image(
        self, image_or_path: Union[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[int]]:
        image, crop = self.tiler.crop_tile(image=self.image, crop=image_or_path)
        return np.moveaxis(image, -1, 0), crop

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, mask, mean, std)
        """
        image, crop = self._read_image(self.images[i])
        mask = self._mask_like_image(image)
        ret = self._apply_transforms(image.astype(np.float32), mask)
        # standardization/normalization step removed since we process the full-sized image
        ret["mean"], ret["std"] = (
            torch.tensor(self.mean).view(1, 1, 1, 1),
            torch.tensor(self.std).view(1, 1, 1, 1),
        )
        ret["crop"] = crop
        return ret
