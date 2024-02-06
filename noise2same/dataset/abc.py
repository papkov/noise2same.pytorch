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
from einops import einops
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

    path: Optional[Union[Path, str]] = None
    mask_percentage: float = 0.5
    pad_divisor: int = 8
    channel_last: bool = True
    standardize: bool = True
    standardize_by_channel: bool = False
    n_dim: int = 2
    n_channels: int = 1
    data_range: int = 255
    n_repeats: int = 1
    input_size: Optional[int] = None
    crop_size: Optional[int] = None
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

    def __str__(self) -> str:
        return 'dataset'

    def _validate(self) -> None:
        """
        Check init arguments types and values
        :return: bool
        """
        pass

    def _validate_path(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            if not self.path.is_dir():
                raise ValueError(
                    f"Incorrect path, {self.path} not a dir. Current working dir: {Path.cwd()}"
                )

    def __post_init__(self) -> None:
        """
        Get a list of images, compose provided transforms with a list of necessary post-transforms
        :return:
        """
        self._validate()
        self._validate_path()

        self.image_index = self._create_image_index()
        if self.transforms is None:
            self.transforms = []
        elif not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        self.transforms = self._compose_transforms(
            self.transforms + self._get_post_transforms(), additional_targets={"ground_truth": "image"}
        )
        for transform in self.transforms:
            if isinstance(transform, (albu.RandomCrop, albu.CenterCrop)):
                self.crop_size = transform.height
                break
            if isinstance(transform, t3d.RandomCrop):
                self.crop_size = transform.patch_size
                break

        # Convert mean and std to torch tensors
        if self.mean is not None and not isinstance(self.mean, torch.Tensor):
            self.mean = einops.rearrange(
                torch.from_numpy(np.array(self.mean).flatten().astype(np.float32)),
                f'n -> n{" 1" * self.n_dim}'
            )
        if self.std is not None and not isinstance(self.std, torch.Tensor):
            self.std = einops.rearrange(
                torch.from_numpy(np.array(self.std).flatten().astype(np.float32)),
                f'n -> n{" 1" * self.n_dim}'
            )

    def __len__(self) -> int:
        return self.get_number_of_images() * self.n_repeats

    def get_number_of_images(self) -> int:
        return len(self.image_index['image'])

    def _compose_transforms(self, *args, **kwargs) -> Union[Compose, t3d.Compose]:
        """
        Compose a list of transforms with a specific function
        :param args:
        :param kwargs:
        :return:
        """
        return Compose(*args, **kwargs) if self.n_dim == 2 else t3d.Compose(*args, **kwargs)

    def _apply_transforms(self, image: Dict[str, Optional[np.ndarray]]) -> Dict[str, T]:
        """
        Apply transforms to both image and mask
        :param image:
        :return:
        """
        return self.transforms(**image)

    def _get_post_transforms(
        self,
    ) -> Union[List[BasicTransform], List[t3d.BaseTransform3D]]:
        """
        Necessary post-transforms (e.g. ToTensor)
        :return:
        """
        return [
            albu.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=self.pad_divisor,
                pad_width_divisor=self.pad_divisor,
            ),
            ToTensorV2(transpose_mask=True)
        ] if self.n_dim == 2 else [t3d.ToTensor(transpose=True)]

    @abstractmethod
    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        """
        Obtain images or their paths from file system
        :return: list of images of paths to them
        """
        raise NotImplementedError

    @abstractmethod
    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        """
        Read a single image from file system or preloaded array
        :param i:
        :return:
        """
        # TODO split to read_image and process_image
        raise NotImplementedError

    def _handle_image(self, image: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for key in filter(lambda k: k != 'crop', image.keys()):
            if image[key].ndim == self.n_dim:
                image[key] = np.expand_dims(image[key], -1)
            image[key] = image[key].astype(np.float32)
        return image

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, mask, mean, std)
        """
        i = i % self.get_number_of_images()
        image = self._get_image(i)
        image = self._handle_image(image)
        image['shape'] = np.array(image['image'].shape) if self.crop_size is None else \
            np.array([self.crop_size] * self.n_dim + [self.n_channels])
        ret = self._apply_transforms(image)
        if self.standardize:
            # by default, self.mean and self.std are None, and normalization is done by patch
            ret["image"], ret["mean"], ret["std"] = self._standardize(ret["image"], self.mean, self.std)
            if 'ground_truth' in image:
                ret["ground_truth"], _, _ = self._standardize(ret["ground_truth"], ret["mean"], ret["std"])
        else:
            # in case the data was normalized or standardized before
            # TODO less ugly way to do this
            ret["mean"] = self.mean if self.mean is not None else torch.tensor(0).view((1,) * ret["image"].ndim)
            ret["std"] = self.std if self.std is not None else torch.tensor(1).view((1,) * ret["image"].ndim)

        # TODO make mask optional
        ret['mask'] = mask_like_image(ret['image'], mask_percentage=self.mask_percentage, channels_last=False)
        return ret

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
class AbstractNoiseDataset3DLarge(AbstractNoiseDataset, ABC):
    """
    For large images where we standardize a full-size image
    """

    input_name: str = None
    tile_size: int = 64
    tile_step: int = 48
    mean: Optional[Union[float, np.ndarray]] = 0
    std: Optional[Union[float, np.ndarray]] = 1
    weight: str = "pyramid"
    channel_last: bool = False
    n_dim: int = 3

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, ground_truth, mask, mean, std, crop)
        """
        image = self._get_image(i)
        image = self._handle_image(image)
        ret = self._apply_transforms(image)
        # standardization/normalization step removed since we process the full-sized image
        ret["mean"], ret["std"] = self.mean, self.std
        # TODO make mask optional
        ret['mask'] = mask_like_image(ret['image'], mask_percentage=self.mask_percentage, channels_last=False)
        return ret

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image, crop = self.tiler.crop_tile(image=self.image, crop=self.image_index['image'][i])
        gt, _ = self.tiler.crop_tile(image=self.ground_truth, crop=crop)
        return {'image': np.moveaxis(image, -1, 0), 'ground_truth': np.moveaxis(gt, -1, 0), 'crop': crop}

    def _read_large_image(self):
        self.image = io.imread(str(self.path / self.input_name)).astype(np.float32)
        # TODO add option to read from disk
        self.ground_truth = self.image.copy()
        assert self.image.shape == self.ground_truth.shape, "Image and ground truth must have the same shape"

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        self._read_large_image()

        if len(self.image.shape) < 4:
            self.image = self.image[..., np.newaxis]
            self.ground_truth = self.ground_truth[..., np.newaxis]

        if self.standardize:
            self.mean = np.mean(self.image, keepdims=True)
            self.std = np.std(self.image, keepdims=True)
            self.image = (self.image - self.mean) / self.std
            self.ground_truth = (self.ground_truth - self.mean) / self.std
        else:
            self.image = normalize_percentile(self.image)
            self.ground_truth = normalize_percentile(self.ground_truth)

        self.tiler = ImageSlicer(
            self.image.shape,
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            weight=self.weight,
            is_channels=True,
        )

        return {'image': self.tiler.crops,
                # TODO make ground_truth_name a data field to read in properly in read_large_image
                'ground_truth': self.tiler.crops}

    def _get_post_transforms(
        self,
    ) -> Union[List[BasicTransform], List[t3d.BaseTransform3D]]:
        """
        Necessary post-transforms (e.g. ToTensor)
        :return:
        """
        return [t3d.ToTensor(transpose=False)]


@dataclass
class AbstractNoiseDataset3DLargeH5(AbstractNoiseDataset3DLarge):
    def _read_large_image(self):
        with h5py.File(str(self.path / self.input_name), "r") as f:
            self.image = np.array(f["image"], dtype=np.float32)
