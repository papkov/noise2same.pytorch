from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor as T

from noise2same.denoiser.abc import Denoiser
from noise2same.denoiser.noise2self import DonutMask


def shuffle(x: T, mask_window_size: int) -> T:
    return rearrange(x, 'b c (h wh) (w ww) -> (wh ww) b c h w', wh=mask_window_size, ww=mask_window_size)


def unshuffle(x: T, mask_window_size: int) -> T:
    return rearrange(x, '(wh ww) b c h w -> b c (h wh) (w ww)', wh=mask_window_size, ww=mask_window_size)


def mask_like(x: T, i: int, mask_window_size: int) -> T:
    """
    Create binary mask where the i-th location in the window is 1
    :param x: input tensor
    :param i: pixel index in the window (e.g. max 15 for 4x4 window)
    :param mask_window_size: size of the window
    :return: interpolated mask tensor
    """
    assert i < mask_window_size ** 2
    mask = torch.zeros_like(x)
    mask = shuffle(mask, mask_window_size)
    mask[i].add_(1)
    mask = unshuffle(mask, mask_window_size)
    return mask


class Blind2Unblind(Denoiser):
    def __init__(
            self,
            n_dim: int = 2,
            in_channels: int = 1,
            mask_window_size: int = 4,
            lambda_rev: float = 2,
            lambda_reg: float = 1,
            **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.in_channels = in_channels
        self.mask_window_size = mask_window_size
        self.lambda_rev = lambda_rev  # re-visibility parameter start value
        self.lambda_reg = lambda_reg  # regularization loss coefficient
        self.donut_filter = DonutMask(n_dim=n_dim, in_channels=in_channels)

    def forward(self, x: T, mask: Optional[T] = None) -> Dict[str, T]:
        """
        Full forward pass of Blind2Unblind denoiser
        (one forward for each pixel in the mask window, one unmasked forward)
        :param x:
        :param mask:
        :return:
        """

        masked = []
        for i in range(self.mask_window_size ** 2):
            x_masked = self.interpolate_mask(x, i)
            out_masked = super().forward(x_masked)['image']
            masked.append(self.shuffle(out_masked)[i])
        masked = self.unshuffle(torch.stack(masked, dim=0))

        with torch.no_grad():
            out = super().forward(x)

        out['image/masked'] = masked
        out['image/combined'] = (out['image/masked'] + out['image'] * self.lambda_rev) / (1 + self.lambda_rev)
        return out

    def compute_loss(self, x_in: Dict[str, T], x_out: Dict[str, T]) -> Tuple[T, Dict[str, float]]:
        diff = x_out['image/masked'] - x_in[self.target_key]
        exp_diff = x_out['image'] - x_in[self.target_key]

        revisible = diff + self.lambda_rev * exp_diff
        loss_reg = self.lambda_reg * torch.mean(diff ** 2)
        loss_rev = torch.mean(revisible ** 2)
        loss = loss_reg + loss_rev

        # loss_rev = self.compute_mse(x_in[self.target_key], x_out['image/combined'])
        # loss_reg = self.compute_mse(x_in[self.target_key], x_out['image/masked'])
        # loss = loss_rev + loss_reg * self.lambda_reg

        loss_dict = {'loss': loss.item(),
                     'loss_rev': loss_rev.item(),
                     'loss_reg': loss_reg.item()}

        return loss, loss_dict

    def interpolate_mask(self, x: T, i: int) -> T:
        """
        Create global mask for the i-th location in the window
        :param x: input tensor
        :param i: pixel index in the window (e.g. max 15 for 4x4 window)
        :return: interpolated mask tensor
        """
        mask = mask_like(x, i, self.mask_window_size)
        filtered = self.donut_filter(x)
        return filtered * mask + x * (1 - mask)

    def shuffle(self, x: T) -> T:
        return shuffle(x, self.mask_window_size)

    def unshuffle(self, x: T) -> T:
        return unshuffle(x, self.mask_window_size)
