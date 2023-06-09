from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy as np
from tqdm.auto import tqdm

from noise2same.dataset.abc import AbstractNoiseDataset2D


@dataclass
class ImagenetDataset(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/ImageNet"
    mode: str = "train"
    version: int = 0  # two noisy copies exist (0, 1)
    standardize_by_channel: bool = True
    n_channels: int = 3

    def _validate(self) -> bool:
        assert self.mode in ("train", "val")
        assert self.version in (0, 1)
        return True

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        data = np.load(self.path / f"{self.mode}.npy", mmap_mode='r')
        return {
            "noisy_input": data[:, self.version + 1],
            "ground_truth": data[:, 0]
        }

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path


@dataclass
class ImagenetTestDataset(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/ImageNet"
    standardize_by_channel: bool = True
    version: int = 0  # for config compatibility

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        return {
            "noisy_input": sorted((self.path / "test").glob("*.npy")),
            "ground_truth": [np.load(p)[0] for p in tqdm(sorted((self.path / "test").glob("*.npy")))]
        }

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return np.load(image_or_path)[1]
