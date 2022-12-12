from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset3D, AbstractNoiseDataset3DLarge
from noise2same.dataset.planaria import PlanariaDatasetPrepared, PlanariaDatasetTiff


@dataclass
class FlyWingDatasetPrepared(PlanariaDatasetPrepared):
    path: Union[Path, str] = "data/Projection_Flywing"
    stack_depth: int = 16

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        depth = image_or_path.shape[1]
        if depth > self.stack_depth:
            # select random set of planes (training)
            start = np.random.choice(depth - self.stack_depth)
            image_or_path = image_or_path[:, start : start + self.stack_depth]
        else:
            # pad with zeros (testing)
            pad = self.stack_depth - depth
            top = pad // 2
            bottom = pad - top
            image_or_path = np.pad(
                image_or_path, ((0, 0), (top, bottom), (0, 0), (0, 0)), "reflect"
            )
        return image_or_path


@dataclass
class FlyWingDatasetTiff(AbstractNoiseDataset3DLarge):
    path: Union[Path, str] = "data/Projection_Flywing/test_data/"
    input_name: str = "C2_T001.tif"
    tile_size: int = 256
    tile_step: int = 192
    crop_border: int = 32
    stack_depth: int = 56
