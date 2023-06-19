from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from torch.nn.functional import conv2d, conv3d

from noise2same.denoiser.abc import Denoiser, DeconvolutionMixin


# TODO factor out
class DonutMask(nn.Module):
    def __init__(self, n_dim: int = 2, in_channels: int = 1):
        """
        Local average excluding the center pixel
        :param n_dim:
        :param in_channels:
        """
        super(DonutMask, self).__init__()
        assert n_dim in (2, 3)
        self.n_dim = n_dim

        kernel = (
            np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
            if n_dim == 2
            else np.array(
                [
                    [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                    [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                    [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                ]
            )
        ).astype(np.float32)
        kernel = kernel / kernel.sum()
        kernel = torch.from_numpy(kernel)[None, None]
        shape = (
            in_channels,
            in_channels,
        ) + (-1,) * n_dim
        kernel = kernel.expand(shape)
        self.register_buffer("kernel", kernel)

    def forward(self, x: T) -> T:
        conv = conv2d if self.n_dim == 2 else conv3d
        return conv(x, self.kernel, padding=1, stride=1)


class Noise2Self(Denoiser):
    """
    Noise2Self denoiser implementation.
    """
    def __init__(
        self,
        n_dim: int = 2,
        in_channels: int = 1,
        masking: str = "gaussian",
        noise_mean: float = 0,
        noise_std: float = 0.2,
        **kwargs: Any,
    ):
        """
        :param n_dim: number of dimensions, either 2 or 3
        :param in_channels: number of input channels
        :param masking: masking type, either 'gaussian' or 'donut'
        :param noise_mean: mean of the gaussian noise for gaussian masking
        :param noise_std: std of the gaussian noise for gaussian masking
        """
        super().__init__(**kwargs)
        assert masking in ("gaussian", "donut")

        self.n_dim = n_dim
        self.in_channels = in_channels
        self.masking = masking
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.mask_kernel = DonutMask(n_dim=self.n_dim, in_channels=self.in_channels)

    def forward(self, x: T, mask: Optional[T] = None) -> Dict[str, T]:
        if mask is not None:
            noise = (
                torch.randn(*x.shape, device=x.device, requires_grad=False) * self.noise_std
                + self.noise_mean
                if self.masking == "gaussian"
                else self.mask_kernel(x)
            )
            x = (1 - mask) * x + mask * noise

        return super().forward(x)

    def compute_loss(self, x_in: Dict[str, T], x_out: Dict[str, T]) -> Tuple[T, Dict[str, float]]:
        loss = self.compute_mse(x_in['image'], x_out['image'], mask=x_in['mask'])
        return loss, {'loss': loss.item(), 'rec_mse': loss.item()}


class Noise2SelfDeconvolution(DeconvolutionMixin, Noise2Self):
    """
    Noise2Self denoiser implementation with deconvolution.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def compute_loss(self, x_in: Dict[str, T], x_out: Dict[str, T]) -> Tuple[T, Dict[str, float]]:
        loss, loss_dict = super().compute_loss(x_in, x_out)
        regularization_loss, regularization_loss_dict = super().compute_regularization_loss(x_in, x_out)
        loss = loss + regularization_loss
        loss_dict.update(regularization_loss_dict)
        loss_dict['loss'] = loss.item()
        return loss, loss_dict
