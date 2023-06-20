from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset


@dataclass
class BSD68Dataset(AbstractNoiseDataset):
    path: Union[Path, str] = "data/BSD68"
    mode: str = "train"

    def __str__(self) -> str:
        return f'bsd68_mixed_{self.mode}'

    def _validate(self) -> None:
        assert self.mode in ("train", "val", "test")

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        path = self.path / self.mode
        files = list(path.glob("*.npy"))
        images = {
            "image": np.load(files[0].as_posix(), allow_pickle=True)
        }
        if self.mode == "test":
            images["ground_truth"] = np.load(files[1].as_posix(), allow_pickle=True)
        return images

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        return {k: v[i] for k, v in self.image_index.items()}
