from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict

import h5py
import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset


def read_h5py(path):
    with h5py.File(path, "r") as f:
        img = np.array(f["image"])
    return img


@dataclass
class HelaShallowDataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/hela"
    channel_id: int = 3
    mode: str = "train"

    def __str__(self) -> str:
        return f'hela_shallow_{self.mode}_ch{self.channel_id}'

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        input_files = list(Path(self.path).glob(f"reconvolved/*ch{self.channel_id}.h5"))
        gt_files = list(Path(self.path).glob(f"deconvolved/*ch{self.channel_id}.h5"))
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

        return {'image': np.stack([read_h5py(f) for f in input_files]),
                'ground_truth': np.stack([read_h5py(f) for f in gt_files])}

    def get_number_of_images(self) -> int:
        return self.image_index['image'].shape[0] * self.image_index['image'].shape[1]

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image_id = i // self.image_index['image'].shape[1]
        plane_id = i % self.image_index['image'].shape[1]
        return {k: v[image_id, plane_id] for k, v in self.image_index.items()}
