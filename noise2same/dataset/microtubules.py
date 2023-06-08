from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from skimage import io

from noise2same.dataset.abc import AbstractNoiseDataset3DLarge
from noise2same.dataset.util import (
    add_microscope_blur_3d,
    add_noise,
    normalize,
)
from noise2same.util import normalize_percentile


@dataclass
class MicrotubulesDataset(AbstractNoiseDataset3DLarge):
    path: Union[Path, str] = "data/microtubules-simulation"
    input_name: str = "input.tif"
    add_blur_and_noise: bool = False

    def _read_large_image(self):
        self.image = io.imread(str(self.path / self.input_name)).astype(np.float32)
        self.ground_truth = normalize_percentile(io.imread(str(self.path / 'ground-truth.tif')).astype(np.float32),
                                                 0.1, 99.9)
        if self.add_blur_and_noise:
            print(f"Generating blur and noise for {self.input_name}")
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
