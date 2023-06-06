from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from torch import Tensor as T

from noise2same.dataset.synthetic import SyntheticDataset


@dataclass
class SyntheticPreparedDataset(SyntheticDataset):
    fixed: bool = False  # if True, read prepared noisy images from disk

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        path_original = self.path / "original"
        path_noisy = self.path / f"noise{self.noise_param}" if self.fixed else path_original
        if not path_noisy.exists():
            print(f"Path {path_noisy} does not exist, generate random images "
                  f"with {self.noise_type} noise {self.noise_param}")
            path_noisy = path_original
            self.fixed = False
        return {"noisy_input": sorted(list(path_noisy.glob(f"*.{self.extension}"))),
                "ground_truth": sorted(list(path_original.glob(f"*.{self.extension}")))}

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


@dataclass
class Set12SyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/Set12"
    extension: str = "png"
    name: str = "set12"


# @dataclass
# class Urban100SyntheticDataset(SyntheticDataset):
#     path: Union[Path, str] = "data/Urban100"
#     extension: str = "png"
#     name: str = "urban100"

