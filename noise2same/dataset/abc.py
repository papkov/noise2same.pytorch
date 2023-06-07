from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as albu
import h5py
import numpy as np
import torch
from albumentations import BasicTransform, Compose
from albumentations.pytorch import ToTensorV2
from pytorch_toolbelt.inference.tiles import ImageSlicer
from skimage import io
from torch import tensor as T
from torch.utils.data import Dataset

from noise2same.dataset import transforms as t3d
from noise2same.dataset.util import mask_like_image
from noise2same.util import normalize_percentile


@dataclass
class AbstractNoiseDataset(Dataset, ABC):
    """
    Abstract noise dataset
    """

    path: Union[Path, str]
    mask_percentage: float = 0.5
    pad_divisor: int = 8
    channel_last: bool = True
    standardize: bool = True
    standardize_by_channel: bool = False
    n_dim: int = 2
    n_channels: int = 1
    mean: Optional[Union[float, np.ndarray]] = None
    std: Optional[Union[float, np.ndarray]] = None
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
        if not self.path.is_dir() and self.path.suffix not in (
            ".tif",
            ".tiff",
        ):
            raise ValueError(
                f"Incorrect path, {self.path} not a dir and {self.path.suffix} is not TIF. "
                f"Current working dir: {Path.cwd()}"
            )

        images = self._get_images()
        self.images = images['noisy_input']
        self.ground_truth = images.get('ground_truth', None)
        if self.transforms is None:
            self.transforms = []
        elif not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        self.transforms = self._compose_transforms(
            self.transforms + self._get_post_transforms(),
            additional_targets={"ground_truth": "image"} if self.ground_truth is not None else None
        )

        # Convert mean and std to torch tensors
        if self.mean is not None and not isinstance(self.mean, torch.Tensor):
            self.mean = torch.from_numpy(self.mean)
        if self.std is not None and not isinstance(self.std, torch.Tensor):
            self.std = torch.from_numpy(self.std)

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
    def _apply_transforms(self, image: np.ndarray, mask: np.ndarray, ground_truth: np.ndarray = None) -> Dict[str, T]:
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
    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
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
        # TODO split to read_image and process_image
        raise NotImplementedError

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, mask, mean, std)
        """
        image = self._read_image(self.images[i]).astype(np.float32)
        # TODO move this functionality to _read_image or a separate method
        if image.ndim == self.n_dim:
            image = np.expand_dims(image, axis=-1 if self.channel_last else 0)

        ground_truth = None
        if self.ground_truth is not None:
            ground_truth = self._read_image(self.ground_truth[i]).astype(np.float32)
            # TODO move this functionality to _read_image or a separate method
            if ground_truth.ndim == self.n_dim:
                ground_truth = np.expand_dims(ground_truth, axis=-1 if self.channel_last else 0)

        mask = self._mask_like_image(image)
        # this was noise_patch in the original code, concatenation does not make any sense
        # https://github.com/divelab/Noise2Same/blob/main/models.py#L154
        # noise_mask = np.concatenate([noise, mask], axis=-1)
        ret = self._apply_transforms(image, mask, ground_truth=ground_truth)
        if self.standardize:
            # by default, self.mean and self.std are None, and normalization is done by patch
            ret["image"], ret["mean"], ret["std"] = self._standardize(ret["image"], self.mean, self.std)
            if self.ground_truth is not None:
                ret["ground_truth"], _, _ = self._standardize(ret["ground_truth"], ret["mean"], ret["std"])
        else:
            # in case the data was normalized or standardized before
            # TODO less ugly way to do this
            ret["mean"] = torch.tensor(0).view((1,) * ret["image"].ndim)
            ret["std"] = torch.tensor(1).view((1,) * ret["image"].ndim)

        return ret

    def _mask_like_image(self, image: np.ndarray) -> np.ndarray:
        return mask_like_image(
            image, mask_percentage=self.mask_percentage, channels_last=self.channel_last
        )

    def _standardize(self, image: T, mean: Optional[T] = None, std: Optional[T] = None) -> Tuple[T, T, T]:
        """
        Normalize an image by mean and std
        :param image: tensor
        :return: normalized image, mean, std
        """
        # Image is already a tensor, hence channel-first
        dim = tuple(range(1, image.ndim))
        if not self.standardize_by_channel:
            dim = (0,) + dim
        # normalize as per the paper
        # TODO in the paper channels are not specified. do they matter? try with dim=(1, 2)
        mean = torch.mean(image, dim=dim, keepdim=True) if mean is None else mean
        std = torch.std(image, dim=dim, keepdim=True) if std is None else std
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
            ToTensorV2(transpose_mask=True)
        ]

    def _apply_transforms(self, image, mask, ground_truth=None) -> Dict[str, T]:
        if ground_truth is None:
            return self.transforms(image=image, mask=mask)
        return self.transforms(image=image, mask=mask, ground_truth=ground_truth)


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

    def _apply_transforms(self, image: np.ndarray, mask: np.ndarray, ground_truth: np.ndarray = None) -> Dict[str, T]:
        ret = {
            "image": self.transforms(image, resample=True),
            "mask": self.transforms(mask, resample=False),
        }
        if ground_truth is not None:
            ret["ground_truth"] = self.transforms(ground_truth, resample=False)
        return ret


@dataclass
class AbstractNoiseDataset3DLarge(AbstractNoiseDataset3D, ABC):
    """
    For large images where we standardize a full-size image
    """

    input_name: str = None
    tile_size: int = 64
    tile_step: int = 48
    mean: float = 0
    std: float = 1
    weight: str = "pyramid"

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, mask, mean, std, crop)
        """
        image, crop = self._read_image(self.images[i])
        mask = self._mask_like_image(image)
        ret = self._apply_transforms(image.astype(np.float32), mask)
        # standardization/normalization step removed since we process the full-sized image
        ret["mean"], ret["std"] = (
            # TODO can rewrite just for self.mean and std?
            torch.tensor(self.mean if self.standardize else 0).view(1, 1, 1, 1),
            torch.tensor(self.std if self.standardize else 1).view(1, 1, 1, 1),
        )
        ret["crop"] = crop
        return ret

    def _read_image(self, image_or_path: List[int]) -> Tuple[np.ndarray, List[int]]:
        image, crop = self.tiler.crop_tile(image=self.image, crop=image_or_path)
        return np.moveaxis(image, -1, 0), crop

    def _read_large_image(self):
        self.image = io.imread(str(self.path / self.input_name)).astype(np.float32)

    def _get_images(self) -> Dict[str, Union[List[str], np.ndarray]]:
        self._read_large_image()

        if len(self.image.shape) < 4:
            self.image = self.image[..., np.newaxis]

        if self.standardize:
            self.mean = self.image.mean()
            self.std = self.image.std()
            self.image = (self.image - self.mean) / self.std
        else:
            self.image = normalize_percentile(self.image)

        self.tiler = ImageSlicer(
            self.image.shape,
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            weight=self.weight,
            is_channels=True,
        )

        return {'noisy_input': self.tiler.crops}


@dataclass
class AbstractNoiseDataset3DLargeH5(AbstractNoiseDataset3DLarge):
    def _read_large_image(self):
        with h5py.File(str(self.path / self.input_name), "r") as f:
            self.image = np.array(f["image"], dtype=np.float32)
