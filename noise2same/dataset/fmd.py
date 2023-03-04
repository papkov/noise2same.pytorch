import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict
import cv2

import numpy as np
from noise2same.dataset.abc import AbstractNoiseDataset2D


@dataclass
class FMDDatasetPrepared(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/FMD"
    mode: str = "train"
    part: str = "cf_fish"

    def _validate(self) -> bool:
        assert self.mode in ("train", "val", "test")
        assert self.part in ("cf_fish", "cf_mice", "tp_mice")
        return True

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
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
                ]) for i in folders
            ] for sub in ('raw', 'gt')
        }
        paths['gt'] = [np.concatenate([folder] * 50) for folder in paths['gt']]
        return {
            "noisy_input": np.concatenate(paths['raw']),
            "ground_truth": np.concatenate(paths['gt'])
        }

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path
