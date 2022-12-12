from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tifffile
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer

from noise2same.dataset.abc import AbstractNoiseDataset3D, AbstractNoiseDataset3DLarge
from noise2same.util import normalize_percentile


@dataclass
class PlanariaDatasetPrepared(AbstractNoiseDataset3D):
    path: Union[Path, str] = "data/Denoising_Planaria"
    mode: str = "train"
    train_size: float = 0.9
    standardize: bool = False  # data was prepared and percentile normalized

    def _get_images(self) -> Union[List[str], np.ndarray]:
        data = np.load(self.path / "train_data/data_label.npz")["X"].astype(np.float32)
        if self.mode == "train":
            data = data[: int(len(data) * self.train_size)]
        else:
            data = data[int(len(data) * self.train_size) :]
        return data

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path


@dataclass
class PlanariaDatasetTiff(AbstractNoiseDataset3DLarge):
    path: Union[Path, str] = "data/Denoising_Planaria/test_data/GT"
    input_name: str = "EXP278_Smed_fixed_RedDot1_sub_5_N7_m0012.tif"
    tile_size: int = 256
    tile_step: int = 192
    crop_border: int = 32
    stack_depth: int = 96
