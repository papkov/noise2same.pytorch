import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy as np
from imageio import imread

from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.util import (
    add_microscope_blur_2d,
    add_noise,
    normalize,
)


@dataclass
class SSIDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/ssi/"
    standardize_by_channel: bool = True
    input_name: str = "drosophila"

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        try:
            files = [f for f in self.path.iterdir() if f.is_file()]
        except FileNotFoundError as e:
            print("File not found, cwd:", os.getcwd())
            raise e

        filename = [f.name for f in files if self.input_name in f.name][0]
        filepath = self.path / filename

        image_clipped = imread(filepath)

        image_clipped = normalize(image_clipped.astype(np.float32))
        blurred_image, psf_kernel = add_microscope_blur_2d(image_clipped)
        noisy_blurred_image = add_noise(
            blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10
        )

        self.psf = psf_kernel

        return {
            "image": noisy_blurred_image[None, ...],
            "ground_truth": image_clipped[None, ...],
        }

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.image_index.items()}
