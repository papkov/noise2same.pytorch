import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Sequence, Tuple
from PIL import Image
import numpy as np
from noise2same.dataset.abc import AbstractNoiseDataset2D


def read_image(path: Union[str, Path]) -> np.ndarray:
    """
    Read image from path with PIL and convert to np.uint8
    :param path: path to image
    :return: np.uint8 image [0, 255]
    """
    im = Image.open(path)
    im = np.array(im, dtype=np.uint8)
    return im


@dataclass
class SyntheticDataset(AbstractNoiseDataset2D):
    noise_type: str = "gaussian"
    extension: str = "JPEG"
    noise_param: Union[int, Tuple[int, int]] = 25
    name: str = ""
    cached: str = ""

    def _validate(self) -> bool:
        assert self.noise_type in ("gaussian", "poisson", "none")
        assert isinstance(self.noise_param, int) or \
               (isinstance(self.noise_param, Sequence) and len(self.noise_param) == 2)
        return True

    def _noise_param(self):
        if isinstance(self.noise_param, int):
            return self.noise_param
        else:
            return np.random.uniform(low=self.noise_param[0], high=self.noise_param[1], size=(1, 1, 1))

    def add_gaussian(self, x: np.ndarray):
        """
        Add gaussian noise to image
        :param x: image [0, 1]

        Adopted from Neighbor2Neighbor https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/2fff2978/train.py#L115
        """
        return np.array(x + np.random.normal(size=x.shape) * self._noise_param() / 255.0, dtype=np.float32)

    def add_poisson(self, x: np.ndarray):
        """
        Add gaussian noise to image
        :param x: image [0, 1]

        Adopted from Neighbor2Neighbor https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/2fff2978/train.py#L124
        """
        lam = self._noise_param()
        return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        if self.cached:
            cached_path = self.path / self.cached
            if cached_path.exists():
                print(f"Cache found in {cached_path}, reading images from npy...\n")
                return {"noisy_input": np.load(self.path / self.cached, allow_pickle=True)}
            else:
                print(f"Cache not found in {cached_path}, read images from disk\n")
        return {"noisy_input": sorted(list(self.path.glob(f"*.{self.extension}")))}

    def add_noise(self, x: np.ndarray):
        if self.noise_type == "gaussian":
            return self.add_gaussian(x)
        elif self.noise_type == "poisson":
            return self.add_poisson(x)
        else:
            return x

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        im = read_image(image_or_path) if not self.cached else image_or_path
        im = im.astype(np.float32) / 255.0
        im = self.add_noise(im)
        return im


@dataclass
class ImagenetSyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/Imagenet_val"
    extension: str = "JPEG"
    name: str = "imagenet"
    cached: str = "Imagenet_val.npy"


@dataclass
class KodakSyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/Kodak"
    extension: str = "png"
    name: str = "kodak"


@dataclass
class Set14SyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/Set14"
    extension: str = "png"
    name: str = "set14"


@dataclass
class BSD300SyntheticDataset(SyntheticDataset):
    path: Union[Path, str] = "data/BSD300/test"
    extension: str = "png"
    name: str = "bsd300"
