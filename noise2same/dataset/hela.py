from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict

import h5py
import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.util import (
    add_microscope_blur_3d,
    add_noise,
    normalize,
)
from noise2same.util import normalize_percentile
import logging

log = logging.getLogger(__name__)


def read_h5py(path):
    with h5py.File(path, "r") as f:
        img = np.array(f["image"], dtype=np.float32)
    return img


@dataclass
class HelaShallowDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/hela/reconvolved"
    channel_id: int = 3
    mode: str = "train"
    add_blur_and_noise: bool = False

    def __str__(self) -> str:
        return f'hela_shallow_{self.mode}_ch{self.channel_id}'

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        self.path = Path(self.path)
        path_gt = self.path.parent / 'deconvolved'
        pattern = f"*ch{self.channel_id}.h5"
        input_files = sorted(list(self.path.glob(pattern)))
        gt_files = sorted(list(path_gt.glob(pattern)))

        assert len(input_files) > 0, f"no files found in {self.path}"
        assert len(input_files) == len(gt_files), f"input and gt files have different lengths"

        if self.mode == "train":
            input_files = input_files[:-1]
            gt_files = gt_files[:-1]
        elif self.mode in ("val", "test"):
            input_files = input_files[-1:]
            gt_files = gt_files[-1:]
        else:
            raise ValueError(f"unknown mode {self.mode}")

        image_index = {'image': [normalize(read_h5py(f)) for f in input_files],
                       'ground_truth': [normalize(read_h5py(f)) for f in gt_files]}

        if self.add_blur_and_noise:
            for i, _ in enumerate(image_index['image']):
                log.info(f"Generating blur and noise for image {i} in {self.mode} stack")
                image_index['image'][i], self.psf = add_microscope_blur_3d(image_index['image'][i], size=17)
                image_index['image'][i] = add_noise(image_index['image'][i],
                                                    alpha=0.001, sigma=0.01, quant_bits=10)

        image_index = {k: np.stack(v) for k, v in image_index.items()}
        return image_index

    def get_number_of_images(self) -> int:
        return self.image_index['image'].shape[0] * self.image_index['image'].shape[1]

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image_id = i // self.image_index['image'].shape[1]
        plane_id = i % self.image_index['image'].shape[1]
        return {k: v[image_id, plane_id] for k, v in self.image_index.items()}


@dataclass
class HelaDataset(HelaShallowDataset):
    n_dim = 3

    def __str__(self) -> str:
        return f'hela_{self.mode}_ch{self.channel_id}'

    def get_number_of_images(self) -> int:
        return self.image_index['image'].shape[0]

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.image_index.items()}
