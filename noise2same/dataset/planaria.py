from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset3D


@dataclass
class PlanariaDatasetPrepared(AbstractNoiseDataset3D):
    path: Union[Path, str] = "data/Denoising_Planaria"
    mode: str = "train"
    train_size: float = 0.9

    def _get_images(self) -> Union[List[str], np.ndarray]:
        data = np.load(self.path / "train_data/data_label.npz")["X"].astype(np.float32)
        if self.mode == "train":
            data = data[: int(len(data) * self.train_size)]
        else:
            data = data[int(len(data) * self.train_size) :]
        return data

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path
