from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from torch import Tensor as T

from noise2same.dataset.synthetic import SyntheticDataset


@dataclass
class BSD400SyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/BSD400"
    extension: str = "png"
    name: str = "bsd400"


@dataclass
class BSD68SyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/BSD68-test/"
    extension: str = "png"
    name: str = "bsd68"
    fixed: bool = False  # if True, read prepared noisy images from disk

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        path = self.path / f"noise{self.noise_param}" if self.fixed else "original"
        assert path.exists(), f"Path {path} does not exist"
        return {"noisy_input": sorted(list(path.glob(f"*.{self.extension}")))}

    def add_noise(self, x: T):
        if self.fixed:
            return x
        return super().add_noise(x)


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

