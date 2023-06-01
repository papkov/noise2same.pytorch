from functools import partial
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as f
from skimage import io
from torch import nn

from noise2same.psf.fft_conv import FFTConv2d, FFTConv3d, fft_conv
from noise2same.util import center_crop


class PSF(nn.Module):
    def __init__(
        self,
        kernel_psf: np.ndarray,
        in_channels: int = 1,
        pad_mode: str = "replicate",
        fft: Union[str, bool] = "auto",
    ):
        """
        Point-spread function
        https://github.com/royerlab/ssi-code/blob/master/ssi/models/psf_convolution.py
        :param kernel_psf: 2D or 3D np.ndarray
        :param pad_mode: {"reflect", "replicate"}
        :param in_channels: int, number of channels to convolve
        :param fft: bool, whether to use FFT convolution
        """
        super().__init__()
        self.kernel_size = kernel_psf.shape[0]
        self.n_dim = len(kernel_psf.shape)
        self.fft = fft
        if self.fft == "auto":
            # TODO run own benchmarks
            # Use FFT Conv if kernel has > 100 elements
            self.fft = self.kernel_size ** self.n_dim > 100
            # self.fft = (self.kernel_size > 21 and self.n_dim == 2) or (
            #     self.kernel_size > 7 and self.n_dim == 3
            # )
        if isinstance(self.fft, str):
            raise ValueError(f"Invalid fft value {self.fft}")

        if self.n_dim == 3 and pad_mode == "reflect":
            # Not supported yet
            pad_mode = "replicate"

        self.pad_mode = pad_mode
        self.pad = (self.kernel_size - 1) // 2
        assert self.n_dim in (2, 3)

        if fft:
            conv = FFTConv2d if self.n_dim == 2 else FFTConv3d
        else:
            conv = nn.Conv2d if self.n_dim == 2 else nn.Conv3d

        self.psf = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            bias=False,
            groups=in_channels,
        )

        self.weights_init(kernel_psf)

    def weights_init(self, kernel_psf: np.ndarray):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel_psf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.pad(x, (self.pad,) * self.n_dim * 2, mode=self.pad_mode)
        return self.psf(x)


class PSFParameter(nn.Module):
    def __init__(
        self,
        kernel_psf: np.ndarray,
        in_channels: int = 1,
        pad_mode: str = "replicate",
        trainable: bool = False,
        fft: Union[str, bool] = "auto",
        auto_padding: bool = False,
    ):
        """
        Parametrized trainable version of PSF
        :param kernel_psf:
        :param in_channels:
        :param pad_mode:
        :param trainable:
        :param auto_padding: (bool) If True, automatically computes padding based on the
                             signal size, kernel size and stride.
        """
        super().__init__()
        self.kernel_size = kernel_psf.shape
        self.n_dim = len(kernel_psf.shape)

        if self.n_dim == 3 and pad_mode == "reflect":
            # Not supported yet
            pad_mode = "replicate"

        self.in_channels = in_channels
        self.pad_mode = pad_mode

        self.pad = [k // 2 for k in self.kernel_size]
        assert self.n_dim in (2, 3)

        self.fft = fft
        if self.fft == "auto":
            # Use FFT Conv if kernel has > 100 elements
            self.fft = np.product(self.kernel_size) > 100
        if isinstance(self.fft, str):
            raise ValueError(f"Invalid fft value {self.fft}")

        if not self.fft:
            auto_padding = False
        self.auto_padding = auto_padding

        self.psf = torch.from_numpy(kernel_psf.squeeze()[(None,) * 2]).float()
        self.psf = nn.Parameter(self.psf, requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.fft:
            conv = partial(
                fft_conv,
                padding_mode=self.pad_mode,
                padding="same" if self.auto_padding else self.pad,
            )
        else:
            signal_padding = tuple(np.repeat(self.pad[::-1], 2))
            x = f.pad(x, signal_padding, mode=self.pad_mode)
            conv = torch.conv2d if self.n_dim == 2 else torch.conv3d

        x = conv(x, self.psf, groups=self.in_channels, stride=1)
        return x


def read_psf(
    path: Union[Path, str], psf_size: Optional[int] = None, normalize: bool = True
) -> np.ndarray:
    """
    Reads PSF from .h5 or .tif file
    :param path: absolute path to file
    :param psf_size: int, optional, crop PSF to a cube of this size if provided
    :param normalize: bool, is divide PSF by its sum
    :return: PSF as numpy array
    """
    path = str(path)
    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            psf = f["psf"]
    else:
        psf = io.imread(path)

    if psf_size is not None:
        psf = center_crop(psf, psf_size)

    if normalize:
        psf /= psf.sum()

    return psf
