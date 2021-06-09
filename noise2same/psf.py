import numpy as np
import torch
from torch import nn


class PSF(nn.Module):
    def __init__(self, kernel_psf: np.ndarray, in_channels: int = 1):
        """
        Point-spread function
        https://github.com/royerlab/ssi-code/blob/master/ssi/models/psf_convolution.py
        :param kernel_psf: 2D or 3D np.ndarray
        :param in_channels: int, number of channels to convolve
        """
        super().__init__()
        self.kernel_size = kernel_psf.shape[0]
        self.n_dim = len(kernel_psf.shape)
        assert self.n_dim in (2, 3)

        pad = nn.ReplicationPad2d if self.n_dim == 2 else nn.ReplicationPad3d
        conv = nn.Conv2d if self.n_dim == 2 else nn.Conv3d

        self.psf = nn.Sequential(
            pad((self.kernel_size - 1) // 2),
            conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                bias=False,
                groups=in_channels,
            ),
        )

        self.weights_init(kernel_psf)

    def weights_init(self, kernel_psf: np.ndarray):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel_psf))

    def forward(self, x):
        return self.psf(x)
