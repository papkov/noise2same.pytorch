from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset2D


@dataclass
class BSD68DatasetPrepared(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/BSD68"
    mode: str = "train"

    def _get_images(self) -> Union[List[str], np.ndarray]:
        path = self.path / self.mode
        files = list(path.glob("*.npy"))
        return np.load(files[0].as_posix(), allow_pickle=True)

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path
