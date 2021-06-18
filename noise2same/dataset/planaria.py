from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset3D


@dataclass
class PlanariaDataset(AbstractNoiseDataset3D):
    path: Union[Path, str] = "data/Denoising_Planaria"
    mode: str = "train"

    def _get_images(self) -> Union[List[str], np.ndarray]:
        pass
