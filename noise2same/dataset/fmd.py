import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import cv2
import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.util import (
    add_microscope_blur_2d,
    add_noise,
    normalize,
)

from tqdm import tqdm


@dataclass
class FMDDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/FMD"
    mode: str = "train"
    part: str = "cf_fish"
    add_blur_and_noise: bool = False

    def _validate(self) -> None:
        assert self.mode in ("train", "val", "test")
        assert self.part in ("cf_fish", "cf_mice", "tp_mice")

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        path = self.path / {
            'cf_fish': 'Confocal_FISH',
            'cf_mice': 'Confocal_MICE',
            'tp_mice': 'TwoPhoton_MICE'
        }[self.part]
        folders = list(range(1, 19)) + [20] if self.mode == 'train' else [19]
        paths = {sub: [
            np.concatenate([
                    np.expand_dims(cv2.imread(str(path / sub / str(i) / image), cv2.IMREAD_GRAYSCALE), 0) for image
                    in sorted(os.listdir(path / sub / str(i))) if image.endswith('png')
                ]) for i in tqdm(folders, desc=sub)
            ] for sub in (('raw', 'gt') if not self.add_blur_and_noise else ('gt', ))
        }
        if self.add_blur_and_noise:
            paths['raw'] = [self._add_blur_and_noise(img[0])[None, ...] for img in tqdm(paths['gt'], desc='blur')]
        else:
            paths['gt'] = [np.concatenate([folder] * 50) for folder in paths['gt']]


        return {
            "noisy_input": np.concatenate(paths['raw']),
            "ground_truth": np.concatenate(paths['gt'])
        }

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image = self.image_index['noisy_input'][i]
        # TODO understand why is it here
        self.mean = np.mean(image, keepdims=True, dtype=np.float32)[None, ...]
        self.std = np.std(image, keepdims=True, dtype=np.float32)[None, ...]
        return {'image': image, 'ground_truth': self.image_index['ground_truth'][i]}

    def _add_blur_and_noise(self, image: np.ndarray) -> np.ndarray:
        image = normalize(image)
        # TODO parametrize
        try:
            image, self.psf = add_microscope_blur_2d(image, size=17)
        except ValueError as e:
            raise ValueError(f"Failed to convolve image {image.shape}") from e
        image = add_noise(
                image,
                alpha=0.001,
                sigma=0.1,
                sap=0,  # 0.01 by default but it is not common to have salt and pepper
                quant_bits=10)
        return image * 255
