from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from torch import Tensor as T

from noise2same.dataset.synthetic import SyntheticDataset, read_image


@dataclass
class SyntheticPreparedDataset(SyntheticDataset):

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        path_original = self.path / "original"
        path_noisy = self.path / f"noise{self.noise_param}" if self.fixed else path_original
        if not path_noisy.exists():
            print(f"Path {path_noisy} does not exist, generate random images "
                  f"with {self.noise_type} noise {self.noise_param}")
            path_noisy = path_original
            self.fixed = False
        return {"image": sorted(list(path_noisy.glob(f"*.{self.extension}"))),
                "ground_truth": sorted(list(path_original.glob(f"*.{self.extension}")))}

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        ret = super()._get_image(i)
        ret['ground_truth'] = read_image(self.image_index['ground_truth'][i])
        return ret

    def add_noise(self, x: T) -> T:
        if self.fixed:
            return x
        return super().add_noise(x)


@dataclass
class BSD400SyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/BSD400"
    extension: str = "png"
    name: str = "bsd400"


@dataclass
class BSD68SyntheticDataset(SyntheticPreparedDataset):
    path: Union[Path, str] = "data/BSD68-test/"
    extension: str = "png"
    name: str = "bsd68"
    n_repeats: int = 4  # 272


@dataclass
class Set12SyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/Set12"
    extension: str = "png"
    name: str = "set12"
    n_repeats: int = 20  # 240


# @dataclass
# class Urban100SyntheticDataset(SyntheticDataset):
#     path: Union[Path, str] = "data/Urban100"
#     extension: str = "png"
#     name: str = "urban100"

