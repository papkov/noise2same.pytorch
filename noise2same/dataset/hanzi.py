from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset


@dataclass
class HanziDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data//Hanzi/tiles/"
    mode: str = "training"
    version: int = 0  # two noisy copies exist (0, 1)
    noise_level: int = 3  # four noise levels (1, 2, 3, 4)

    def _validate(self) -> None:
        assert self.mode in ("training", "testing", "validation")
        assert self.noise_level in (1, 2, 3, 4)
        assert self.version in (0, 1)

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        data = np.load(self.path / f"{self.mode}.npy", mmap_mode='r')
        return {
            "image": data[:, self.version * 4 + self.noise_level],
            "ground_truth": data[:, 0]
        }

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.image_index.items()}
