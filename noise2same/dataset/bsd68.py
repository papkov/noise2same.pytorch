from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset


@dataclass
class BSD68Dataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/BSD68"
    mode: str = "train"

    def _validate(self) -> None:
        assert self.mode in ("train", "val", "test")

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        path = self.path / self.mode
        files = list(path.glob("*.npy"))
        images = {
            "noisy_input": np.load(files[0].as_posix(), allow_pickle=True)
        }
        if self.mode == "test":
            images["ground_truth"] = np.load(files[1].as_posix(), allow_pickle=True)
        return images

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path
