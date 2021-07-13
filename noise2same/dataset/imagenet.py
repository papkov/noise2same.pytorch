from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset2D


@dataclass
class ImagenetDatasetPrepared(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/ImageNet"
    mode: str = "train"
    version: int = 0  # two noisy copies exist (0, 1)
    standardize_by_channel: bool = True

    def _validate(self) -> bool:
        assert self.mode in ("train", "val")
        assert self.version in (0, 1)
        return True

    def _get_images(self) -> Union[List[str], np.ndarray]:
        return np.load(self.path / f"{self.mode}.npy")[:, self.version + 1]

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path


@dataclass
class ImagenetDatasetTest(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/ImageNet"
    standardize_by_channel: bool = True

    def _get_images(self) -> Union[List[str], np.ndarray]:
        return sorted((self.path / "test").glob("*.npy"))

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return np.load(image_or_path)[1]
