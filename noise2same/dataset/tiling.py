from dataclasses import dataclass
from typing import Dict, Union, List, Tuple, Any, Iterable

from torch.utils.data import DataLoader
from torch import Tensor as T
import torch
import numpy as np

from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from noise2same.dataset.abc import AbstractNoiseDataset


@dataclass
class TiledImageFactory:
    tile_size: Union[int, Tuple[int]]
    tile_step: Union[int, Tuple[int]]
    crop_border: Union[int, Tuple[int]]
    weight: str = 'pyramid'
    batch_size: int = 1
    num_workers: int = 8

    def produce(self, image: Dict[str, Union[np.ndarray, T]]) -> Tuple[DataLoader, TileMerger]:
        self.tile_size, self.tile_step, self.crop_border = map(
            lambda x: tuple(x) if isinstance(x, Iterable) else x, (self.tile_size, self.tile_step, self.crop_border)
        )
        dataset = TiledImageDataset(
            image=image['image'],
            mean=image['mean'],
            std=image['std'],
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            crop_border=self.crop_border,
            weight=self.weight,
            n_channels=image['image'].shape[0],
            n_dim=len(image['shape']) - 1
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
        )
        return loader, merger


@dataclass
class TiledImageDataset(AbstractNoiseDataset):
    tile_size: Union[int, Tuple[int]] = 256
    tile_step: Union[int, Tuple[int]] = 192
    crop_border: Union[int, Tuple[int]] = 0
    weight: str = 'pyramid'
    image: Union[np.ndarray, T] = None

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        :param i: int, index
        :return: dict(image, mask, mean, std, crop)
        """
        image, crop = self._get_image(i)
        image = self._handle_image(image)
        image['mask'] = self._mask_like_image(image['image'])
        ret = self._apply_transforms(image)
        # standardization/normalization step removed since we process the full-sized image
        ret["mean"], ret["std"] = (
            # TODO can rewrite just for self.mean and std?
            torch.tensor(self.mean if self.standardize else 0),
            torch.tensor(self.std if self.standardize else 1),
        )
        ret["crop"] = crop
        return ret

    def _create_image_index(self) -> Dict[str, Union[List[str], np.ndarray]]:
        assert self.image is not None
        if self.image.shape[0] == self.n_channels:
            self.image = torch.moveaxis(self.image, 0, -1)
        self.slicer = ImageSlicer(
            self.image.shape,
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            weight=self.weight,
            is_channels=True,
            crop_border=self.crop_border
        )
        return {'image': self.slicer.crops}

    def _get_image(self, i: int) -> Tuple[Dict[str, np.ndarray], List[int]]:
        image, crop = self.slicer.crop_tile(image=self.image, crop=self.image_index['image'][i])
        return {'image': image}, crop
