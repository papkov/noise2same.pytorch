from itertools import product
from typing import List, Tuple

import albumentations as albu
import numpy as np
from albumentations import Compose


def get_stratified_coords(
    box_size: int, shape: Tuple[int, ...]
) -> Tuple[List[int], ...]:
    """
    Create stratified blind spot coordinates
    :param box_size: int, size of stratification box
    :param shape: tuple, image shape
    :return:
    """
    box_count = [int(np.ceil(s / box_size)) for s in shape]
    coords = []

    for ic in product(*[np.arange(bc) for bc in box_count]):
        coord = tuple(np.random.rand() * box_size for _ in shape)
        coord = [int(i * box_size + c) for i, c in zip(ic, coord)]
        if all(c < s for c, s in zip(coord, shape)):
            coords.append(coord)

    coords = tuple(zip(*coords))  # transpose (N, 3) -> (3, N)
    return coords


def mask_like_image(
    image: np.ndarray, mask_percentage: float = 0.5, channels_last: bool = True
) -> np.ndarray:
    """
    Generates a stratified mask of image.shape
    :param image: ndarray, reference image to mask
    :param mask_percentage: float, percentage of pixels to mask, default 0.5%
    :param channels_last: bool, true to process image as channel-last (256, 256, 3)
    :return: ndarray, mask
    """
    # todo understand generator_val
    # https://github.com/divelab/Noise2Same/blob/8cdbfef5c475b9f999dcb1a942649af7026c887b/models.py#L130
    mask = np.zeros_like(image)
    n_channels = image.shape[-1 if channels_last else 0]
    channel_shape = image.shape[:-1] if channels_last else image.shape[1:]
    for c in range(n_channels):
        box_size = np.round(np.sqrt(100 / mask_percentage)).astype(np.int)
        mask_coords = get_stratified_coords(box_size=box_size, shape=channel_shape)
        mask_coords = (mask_coords + (c,)) if channels_last else ((c,) + mask_coords)
        mask[mask_coords] = 1.0
    return mask


def training_augmentations_2d(crop: int = 64):
    return Compose(
        [
            albu.RandomCrop(width=crop, height=crop, p=1),
            albu.RandomRotate90(p=0.5),
            albu.Flip(p=0.5),
        ]
    )
