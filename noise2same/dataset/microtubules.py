from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
from skimage import io

from noise2same.dataset.abc import AbstractNoiseDataset, AbstractNoiseDataset3DLarge
from noise2same.dataset.util import (
    add_microscope_blur_3d,
    add_noise,
    normalize,
)
from noise2same.util import normalize_percentile
import logging

log = logging.getLogger(__name__)


@dataclass
class MicrotubulesDataset(AbstractNoiseDataset3DLarge):
    path: Union[Path, str] = "data/microtubules-simulation"
    input_name: str = "input.tif"
    add_blur_and_noise: bool = False

    def __str__(self) -> str:
        return f'microtubules_train_{self.input_name}'

    def _read_large_image(self):
        self.image = io.imread(str(self.path / self.input_name)).astype(np.float32)
        self.ground_truth = normalize_percentile(io.imread(str(self.path / 'ground-truth.tif')).astype(np.float32),
                                                 0.1, 99.9)
        if self.add_blur_and_noise:
            log.info(f"Generating blur and noise for {self.input_name}")
            # self.image = normalize_percentile(self.image, 0.1, 99.9)
            self.image = normalize(self.image)
            # TODO parametrize
            self.image, self.psf = add_microscope_blur_3d(self.image, size=17)
            self.image = add_noise(
                self.image,
                alpha=0.001,
                sigma=0.1,
                sap=0,  # 0.01 by default but it is not common to have salt and pepper
                quant_bits=10,
            )


@dataclass
class MicrotubulesTestDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/microtubules-simulation"
    input_name: str = "input.tif"
    add_blur_and_noise: bool = False

    def __str__(self) -> str:
        return f'microtubules_test_{self.input_name}'

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        # TODO remove duplicate code
        image = io.imread(str(self.path / self.input_name)).astype(np.float32)
        ground_truth = normalize_percentile(io.imread(str(self.path / 'ground-truth.tif')).astype(np.float32),
                                            0.1, 99.9)
        if self.add_blur_and_noise:
            log.info(f"Generating blur and noise for {self.input_name}")
            # self.image = normalize_percentile(self.image, 0.1, 99.9)
            image = normalize(image)
            # TODO parametrize
            image, self.psf = add_microscope_blur_3d(image, size=17)
            image = add_noise(
                image,
                alpha=0.001,
                sigma=0.1,
                sap=0,  # 0.01 by default but it is not common to have salt and pepper
                quant_bits=10,
            )

        return {
            'image': [image],
            'ground_truth': [ground_truth],
        }

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.image_index.items()}
