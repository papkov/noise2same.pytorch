from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset


@dataclass
class ImagenetDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/ImageNet"
    mode: str = "train"
    version: int = 0  # two noisy copies exist (0, 1)
    standardize_by_channel: bool = True
    n_channels: int = 3

    def _validate(self) -> None:
        assert self.mode in ("train", "val")
        assert self.version in (0, 1)

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        data = np.load(self.path / f"{self.mode}.npy", mmap_mode='r')
        return {
            "noisy_input": data[:, self.version + 1],
            "ground_truth": data[:, 0]
        }

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {'image': self.image_index['noisy_input'][i], 'ground_truth': self.image_index['ground_truth'][i]}


@dataclass
class ImagenetTestDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/ImageNet"
    standardize_by_channel: bool = True
    version: int = 0  # for config compatibility

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        return {
            "paths": sorted((self.path / "test").glob("*.npy")),
        }

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        pair = np.load(self.image_index['paths'][i])
        return {'image': pair[1], 'ground_truth': pair[0]}
