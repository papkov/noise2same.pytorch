from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as albu
import numpy as np
import torch
from albumentations import BasicTransform, Compose
from albumentations.pytorch import ToTensorV2
from torch import tensor as T
from torch.utils.data import Dataset

from noise2same.dataset import transforms as t3d
from noise2same.dataset.util import mask_like_image


@dataclass
class AbstractNoiseDataset(Dataset, ABC):
    """
    Abstract noise dataset
    """

    path: Union[Path, str]
    mask_percentage: float = 0.5
    pad_divisor: int = 8
    channel_last: bool = True
    normalize_by_channel: bool = False
    n_dim: int = 2
    transforms: Optional[
        Union[
            List[BasicTransform],
            Compose,
            List[Compose],
            List[t3d.BaseTransform3D],
            t3d.Compose,
            List[t3d.Compose],
        ]
    ] = None

    def _validate(self) -> bool:
        """
        Check init arguments types and values
        :return: bool
        """
        return True

    def __post_init__(self) -> None:
        """
        Get a list of images, compose provided transforms with a list of necessary post-transforms
        :return:
        """
        if not self._validate():
            raise ValueError("Validation failed")

        self.path = Path(self.path)
        assert self.path.is_dir(), f"Incorrect path, {self.path} not a dir"

        self.images = self._get_images()
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        self.transforms = self._compose_transforms(
            self.transforms + self._get_post_transforms()
        )

    def __len__(self) -> int:
        return len(self.images)

    @abstractmethod
    def _compose_transforms(self, *args, **kwargs) -> Union[Compose, t3d.Compose]:
        """
        Compose a list of transforms with a specific function
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _apply_transforms(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, T]:
        """
        Apply transforms to both image and mask
        :param image:
        :param mask:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _get_post_transforms(
        self,
    ) -> Union[List[BasicTransform], List[t3d.BaseTransform3D]]:
        """
        Necessary post-transforms (e.g. ToTensor)
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _get_images(self) -> Union[List[str], np.ndarray]:
        """
        Obtain images or their paths from file system
        :return: list of images of paths to them
        """
        raise NotImplementedError

    @abstractmethod
    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Read a single image from file system or preloaded array
        :param image_or_path:
        :return:
        """
        raise NotImplementedError

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, mask, mean, std)
        """
        image = self._read_image(self.images[i]).astype(np.float32)
        if image.ndim == self.n_dim:
            image = np.expand_dims(image, axis=-1 if self.channel_last else 0)

        mask = self._mask_like_image(image)
        # this was noise_patch in the original code, concatenation does not make any sense
        # https://github.com/divelab/Noise2Same/blob/main/models.py#L154
        # noise_mask = np.concatenate([noise, mask], axis=-1)
        ret = self._apply_transforms(image, mask)
        ret["image"], ret["mean"], ret["std"] = self._normalize(ret["image"])
        return ret

    def _mask_like_image(self, image: np.ndarray) -> np.ndarray:
        return mask_like_image(
            image, mask_percentage=self.mask_percentage, channels_last=self.channel_last
        )

    def _normalize(self, image: T) -> Tuple[T, T, T]:
        """
        Normalize an image by mean and std
        :param image: tensor
        :return: normalized image, mean, std
        """
        # Image is already a tensor, hence channel-first
        dim = tuple(range(1, image.ndim))
        if not self.normalize_by_channel:
            dim = (0,) + dim
        # normalize as per the paper
        # TODO in the paper channels are not specified. do they matter? try with dim=(1, 2)
        mean = torch.mean(image, dim=dim, keepdim=True)
        std = torch.std(image, dim=dim, keepdim=True)
        image = (image - mean) / std
        return image, mean, std


@dataclass
class AbstractNoiseDataset2D(AbstractNoiseDataset, ABC):
    def _compose_transforms(self, *args, **kwargs) -> Compose:
        return Compose(*args, **kwargs)

    def _get_post_transforms(self) -> List[BasicTransform]:
        return [
            albu.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=self.pad_divisor,
                pad_width_divisor=self.pad_divisor,
            ),
            ToTensorV2(transpose_mask=True),
        ]

    def _apply_transforms(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, T]:
        return self.transforms(image=image, mask=mask)


@dataclass
class AbstractNoiseDataset3D(AbstractNoiseDataset, ABC):
    channel_last: bool = False
    n_dim: int = 3
    transforms: Optional[
        Union[List[t3d.BaseTransform3D], t3d.Compose, List[t3d.Compose]]
    ] = None

    def _compose_transforms(self, *args, **kwargs) -> t3d.Compose:
        return t3d.Compose(*args, **kwargs)

    def _get_post_transforms(self) -> List[t3d.BaseTransform3D]:
        return [t3d.ToTensor(transpose=False)]

    def _apply_transforms(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, T]:
        ret = {
            "image": self.transforms(image, resample=True),
            "mask": self.transforms(mask, resample=False),
        }
        return ret
