from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor as T

Ints = Union[int, Tuple[int, ...], List[int]]
Array = Union[ndarray, T]


@dataclass
class BaseTransform3D(ABC):
    p: float = 0.5
    axis: Ints = 0
    seed: int = 43
    k: int = 0
    done: bool = False
    channel_axis: Optional[Ints] = None

    def __post_init__(self):
        np.random.seed(self.seed)

    @abstractmethod
    def apply(self, x: ndarray) -> ndarray:
        raise NotImplementedError

    @abstractmethod
    def resample(self, x: ndarray) -> None:
        raise NotImplementedError

    def __call__(self, x: Array, resample: bool = False) -> Array:
        # If we do not resample, check if transform was done
        if not resample:
            if self.done:
                # if transform was applied before and we do not resample, always transform
                return self.apply(x)
            else:
                return x

        # If we resample
        else:
            self.resample(x)
            if np.random.uniform() < self.p:
                # transform with probability p
                self.done = True
                return self.apply(x)
            else:
                # otherwise return identity
                self.done = False
                return x


class RandomFlip(BaseTransform3D):
    def resample(self, x: ndarray) -> None:
        dims = np.arange(x.ndim)
        dims[-1] = -1
        if self.channel_axis is not None:
            dims = np.delete(dims, self.channel_axis)
        self.axis = np.random.choice(dims)

    def apply(self, x: ndarray) -> ndarray:
        # .copy() solves negative stride issue
        return np.flip(x, axis=self.axis).copy()


class RandomRotate90(BaseTransform3D):
    axis: Tuple[int, int] = (0, 1)

    def apply(self, x: ndarray) -> ndarray:
        return np.rot90(x, k=self.k, axes=self.axis).copy()

    def resample(self, x: ndarray) -> None:
        dims = np.arange(x.ndim)
        dims[-1] = -1
        if self.channel_axis is not None:
            dims = np.delete(dims, self.channel_axis)
        self.k = np.random.choice(4)
        a = int(np.random.choice(len(dims)))
        self.axis = (dims[a], dims[a - 1])


class RandomCrop(BaseTransform3D):
    p: float = 1
    patch_size: Union[None, int, Tuple[int, ...]] = 64
    start: Optional[Union[int, List[int]]] = None

    def patch_tuple(self, x: Array) -> Tuple[int, ...]:
        """
        Forms a correct tuple of patch shape from provided init argument `patch_size`
        :param x: array to crop a patch from
        :return: tuple with patch size
        """
        if self.patch_size is None:
            # crop patch half a size of the original if None
            return tuple(
                s // 2 if s != 1 and i != self.channel_axis else None
                for i, s in enumerate(x.shape)
            )
        if isinstance(self.patch_size, int):
            return tuple(
                self.patch_size if s != 1 and i != self.channel_axis else None
                for i, s in enumerate(x.shape)
            )
        else:
            assert len(self.patch_size) == x.squeeze().n_dim
            return self.patch_size

    def slice(self, x: Array) -> Tuple[slice, ...]:
        """
        Returns tuple of slices to slice a given array
        :param x: array to slice
        :return: tuple of slices
        """
        patch_size = self.patch_tuple(x)

        # Create slices from patch_size and start points
        slices = tuple(
            slice(s, s + p) if p is not None else slice(None)
            for s, p in zip(self.start, patch_size)
        )

        return slices

    def resample(self, x: ndarray) -> None:
        patch_size = self.patch_tuple(x)
        self.start = [
            np.random.choice(s - p) if p is not None else p
            for s, p in zip(x.shape, patch_size)
        ]

    def apply(self, x: Array) -> Array:
        s = self.slice(x)
        return x[s]


class CenterCrop(RandomCrop):
    def slice(self, x: Array) -> Tuple[slice, ...]:
        patch_size = self.patch_tuple(x)
        center = [s // 2 if p is not None else p for s, p in zip(x.shape, patch_size)]
        return tuple(
            slice(c - int(np.floor(p / 2)), c + int(np.ceil(p / 2)))
            if p is not None
            else slice(None)
            for c, p in zip(center, patch_size)
        )


@dataclass
class Compose:
    transforms: List[BaseTransform3D]
    debug: bool = False

    def __call__(self, x: ndarray, resample: bool = False):
        out = x.copy()
        for t in self.transforms:
            if t is not None:
                if self.debug:
                    print(f"Apply {t}")
                out = t(out, resample=resample)
        return out


@dataclass
class ToTensor(BaseTransform3D):
    transpose: bool = False
    p: int = 1
    done: bool = True

    def resample(self, x: ndarray) -> None:
        self.done = True

    def apply(self, x: ndarray) -> T:
        out = x.copy()
        if self.transpose:
            out = np.moveaxis(out, -1, 0)
        out = torch.from_numpy(out)
        return out
