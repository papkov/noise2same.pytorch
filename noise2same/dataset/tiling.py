from dataclasses import dataclass
from typing import Dict, Union, List, Tuple, Any, Iterable, Optional

import numpy as np
import torch
from albumentations import BasicTransform, Compose
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from torch import Tensor as T
from torch.utils.data import DataLoader

from noise2same.dataset import transforms as t3d
from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.util import mask_like_image


@dataclass
class TiledImageFactory:
    tile_size: Union[int, Tuple[int]]
    tile_step: Union[int, Tuple[int]]
    crop_border: Union[int, Tuple[int]]
    weight: str = 'pyramid'
    batch_size: int = 1
    num_workers: int = 8
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

    def produce(self,
                image: Dict[str, Union[np.ndarray, T]],
                keys: Optional[Tuple[str, ...]] = None) -> Tuple[DataLoader, TileMerger]:
        self.tile_size, self.tile_step, self.crop_border = map(
            lambda x: tuple(x) if isinstance(x, Iterable) else x, (self.tile_size, self.tile_step, self.crop_border)
        )
        """
        Produce a data loader for tiled dataset and a merger to combine tiles back into image
        :param image: dict with 'image' to tile and its parameters ('shape', 'mean', 'std')
        :param keys: keys to use for merging the output
        """
        dataset = TiledImageDataset(
            image=image['image'],
            ground_truth=image.get('ground_truth'),
            mean=image['mean'],
            std=image['std'],
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            crop_border=self.crop_border,
            weight=self.weight,
            n_channels=image['image'].shape[0],
            n_dim=len(image['shape']) - 1,
            transforms=self.transforms
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        merger = TileMerger(
            image_shape=dataset.slicer.target_shape,
            channels=image['image'].shape[0],
            weight=dataset.slicer.weight,
            device=image['image'].device,
            crop_border=self.crop_border if isinstance(self.crop_border, int) else self.crop_border[0],
            default_value=0,
            keys=keys,
        )
        return loader, merger


@dataclass
class TiledImageDataset(AbstractNoiseDataset):
    tile_size: Union[int, Tuple[int]] = 256
    tile_step: Union[int, Tuple[int]] = 192
    crop_border: Union[int, Tuple[int]] = 0
    weight: str = 'pyramid'
    standardize: bool = False
    image: Union[np.ndarray, T] = None
    ground_truth: Union[np.ndarray, T] = None

    def __str__(self) -> str:
        return f'tiled_image_dataset'

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        assert self.image is not None
        if self.image.shape[0] == self.n_channels:
            self.image = torch.moveaxis(self.image, 0, -1)
            if self.ground_truth is not None:
                self.ground_truth = torch.moveaxis(self.ground_truth, 0, -1)
        self.slicer = ImageSlicer(
            self.image.shape,
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            weight=self.weight,
            is_channels=True,
            crop_border=self.crop_border
        )
        return {'image': self.slicer.crops}

    def _get_image(self, i: int) -> Dict[str, np.ndarray]:
        image, crop = self.slicer.crop_tile(image=self.image, crop=self.image_index['image'][i])
        if self.ground_truth is not None:
            ground_truth, _ = self.slicer.crop_tile(image=self.ground_truth, crop=crop)
            return {'image': image, 'ground_truth': ground_truth, 'crop': crop}
        return {'image': image, 'crop': crop}
