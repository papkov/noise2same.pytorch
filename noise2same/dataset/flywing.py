from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from noise2same.dataset.planaria import PlanariaDatasetPrepared, PlanariaDatasetTiff


@dataclass
class FlyWingDatasetPrepared(PlanariaDatasetPrepared):
    path: Union[Path, str] = "data/‘Projection_Flywing"

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        # pad stack depth 50 to 64
        image_or_path = np.pad(
            image_or_path, ((0, 0), (7, 7), (0, 0), (0, 0)), "reflect"
        )
        return image_or_path


@dataclass
class FlyWingDatasetTiff(PlanariaDatasetTiff):
    stack_depth: int = 64
