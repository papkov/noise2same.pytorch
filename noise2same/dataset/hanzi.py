from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset2D


@dataclass
class HanziDataset(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data//Hanzi/tiles/"
    mode: str = "training"
    version: int = 0  # two noisy copies exist (0, 1)
    noise_level: int = 3  # four noise levels (1, 2, 3, 4)

    def _validate(self) -> bool:
        assert self.mode in ("training", "testing", "validation")
        assert self.noise_level in (1, 2, 3, 4)
        assert self.version in (0, 1)
        return True

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        data = np.load(self.path / f"{self.mode}.npy")
        return {
            "noisy_input": data[:, self.version * 4 + self.noise_level],
            "ground_truth": data[:, 0]
        }

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path
