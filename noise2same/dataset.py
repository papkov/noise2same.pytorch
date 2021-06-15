from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import albumentations
import albumentations as albu
import numpy as np
import torch
from albumentations import BasicTransform, Compose, Normalize
from albumentations.pytorch import ToTensorV2
from torch import Tensor as T
from torch.utils.data import Dataset


def get_stratified_coords(
    box_size: int, shape: Tuple[int, ...]
) -> Tuple[List[int], ...]:
    box_count = [int(np.ceil(s / box_size)) for s in shape]
    coords = []

    for ic in product(*[np.arange(bc) for bc in box_count]):
        coord = tuple(np.random.rand() * box_size for _ in shape)
        coord = [int(i * box_size + c) for i, c in zip(ic, coord)]
        if all(c < s for c, s in zip(coord, shape)):
            coords.append(coord)

    coords = tuple(zip(*coords))  # transpose (N, 3) -> (3, N)
    return coords


def mask_like_image(image: np.ndarray, mask_percentage: float = 0.5) -> np.ndarray:
    """
    Generates a stratified mask of image.shape
    :param image:
    :param mask_percentage:
    :return:
    """
    # todo understand generator_val
    # https://github.com/divelab/Noise2Same/blob/8cdbfef5c475b9f999dcb1a942649af7026c887b/models.py#L130
    mask = np.zeros_like(image)
    for c in range(image.shape[-1]):
        box_size = np.round(np.sqrt(100 / mask_percentage)).astype(np.int)
        mask_coords = get_stratified_coords(box_size=box_size, shape=image.shape[:-1])
        mask[mask_coords + (c,)] = 1.0
    return mask


@dataclass
class AbstractNoiseDataset2D(Dataset):
    path: Union[Path, str]
    mask_percentage: float = 0.5
    # mean: Tuple[int, ...] = (0.485, 0.456, 0.406)
    # std: Tuple[int, ...] = (0.229, 0.224, 0.225)
    transforms: Optional[Union[List[BasicTransform], Compose, List[Compose]]] = None

    def __post_init__(self):
        self.path = Path(self.path)
        assert self.path.is_dir(), f"Incorrect path, {self.path} not a dir"

        self.images = self._get_images()
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        self.transforms = Compose(
            self.transforms
            + [
                albu.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=32,
                    pad_width_divisor=32,
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def _get_images(self) -> Union[List[str], np.ndarray]:
        raise NotImplementedError

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, mask, mean, std)
        """
        image = self._read_image(self.images[i]).astype(np.float32)
        if image.ndim == 2:
            image = image[..., np.newaxis]

        mask = mask_like_image(image, self.mask_percentage)
        # this was noise_patch in the original code, concatenation does not make any sense
        # https://github.com/divelab/Noise2Same/blob/main/models.py#L154
        # noise_mask = np.concatenate([noise, mask], axis=-1)
        ret = self.transforms(image=image, mask=mask)

        # normalize as per the paper
        # TODO in the paper channels are not specified. do they matter? try with dim=(1, 2)
        mean = torch.mean(ret["image"], dim=(0, 1, 2), keepdim=True)
        std = torch.std(ret["image"], dim=(0, 1, 2), keepdim=True)
        ret["image"] = (ret["image"] - mean) / std

        ret.update({"mean": mean, "std": std})
        return ret


@dataclass
class BSD68DatasetPrepared(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/BSD68"
    mode: str = "train"

    def _get_images(self) -> Union[List[str], np.ndarray]:
        path = Path(self.path) / self.mode
        files = list(path.glob("*.npy"))
        return np.load(files[0].as_posix(), allow_pickle=True)

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path
