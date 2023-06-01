from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F, Identity
from torch.nn.functional import conv2d, conv3d
from torchvision.transforms import GaussianBlur

from noise2same.backbone.swinia import SwinIA
from noise2same.backbone.unet import UNet, RegressionHead
from noise2same.backbone.swinir import SwinIR
from noise2same.psf.psf_convolution import PSFParameter, read_psf


class ModelMode(Enum):

    NOISE2TRUE = 0,
    NOISE2SELF = 1,
    NOISE2SAME = 2


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
        )
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
        # todo understand stride
        return conv(x, self.kernel, padding=1, stride=1)


class Noise2Same(nn.Module):
    def __init__(
        self,
        n_dim: int = 2,
        in_channels: int = 1,
        base_channels: int = 96,
        lambda_bsp: float = 0,
        lambda_rec: float = 1.0,
        lambda_inv: float = 2.0,
        lambda_inv_deconv: float = 0.0,
        masked_inv_deconv: bool = True,
        mask_percentage: float = 0.5,
        masking: str = "gaussian",
        noise_mean: float = 0,
        noise_std: float = 0.2,
        lambda_bound: float = 0,
        lambda_sharp: float = 0,
        psf: Optional[Union[str, np.ndarray]] = None,
        psf_size: Optional[int] = None,
        psf_pad_mode: str = "reflect",
        residual: bool = False,
        regularization_key: str = "image",
        mode: str = "noise2same",
        psf_fft: Union[str, bool] = "auto",
        backbone: Union[UNet, SwinIR, Identity] = Identity(),
        head: Union[RegressionHead, Identity] = Identity(),
        **kwargs: Any,
    ):
        """

        :param n_dim:
        :param in_channels:
        :param base_channels:
        :param lambda_inv:
        :param mask_percentage:
        :param masking:
        :param noise_mean:
        :param noise_std:
        :param psf:
        """
        super(Noise2Same, self).__init__()
        assert masking in ("gaussian", "donut")
        assert regularization_key in ("image", "deconv")
        if psf is None:
            # we don't have a psf, so we can't use deconv
            regularization_key = "image"

        self.n_dim = n_dim
        self.in_channels = in_channels
        self.lambda_bsp = lambda_bsp
        self.lambda_rec = lambda_rec
        self.lambda_inv = lambda_inv
        self.lambda_inv_deconv = lambda_inv_deconv
        self.masked_inv_deconv = masked_inv_deconv
        self.mask_percentage = mask_percentage
        self.masking = masking
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.residual = residual
        self.regularization_key = regularization_key
        self.lambda_bound = lambda_bound
        self.lambda_sharp = lambda_sharp
        self.mode = ModelMode[mode.upper()]

        # TODO customize with segmentation_models
        self.net = backbone
        self.head = head

        # todo parametrize
        self.blur = GaussianBlur(5, sigma=0.2) if residual else None

        self.mask_kernel = DonutMask(n_dim=n_dim, in_channels=in_channels)

        self.psf = None
        if psf is not None:
            if isinstance(psf, (str, Path)):
                # read by path, otherwise assume ndarray
                # TODO check
                psf = read_psf(psf, psf_size=psf_size)
            print("PSF shape", psf.shape)
            self.psf = PSFParameter(psf, pad_mode=psf_pad_mode, fft=psf_fft)
            for param in self.psf.parameters():
                param.requires_grad = False

    def forward(
        self,
        x: T,
        mask: T = None,
        convolve: bool = True,
        crops: Optional[T] = None,
        full_size_image: Optional[T] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Union[Dict[str, T], None], Union[Dict[str, T], None]]:
        """
        Make two forward passes: with mask and without mask
        :param x:
        :param mask:
        :param convolve: if True, convolve the output with the PSF
        :param crops: if not None, positions of crops `x` in full size image. NB! we assume that crops do not overlap
        :param full_size_image: if not None, full size image from which `x` was taken by `crops` coordinates
        :return: tuple of dictionaries of tensors (output for masked input, output for raw input):
                    image - final output, always present
                    deconv - output before PSF if PSF is provided and `convolve` is True
        """
        out_raw, out_mask = None, None
        if self.mode != ModelMode.NOISE2SELF or mask is None:
            out_raw = self.forward_whole(x, convolve, crops, full_size_image)
        if mask is not None:
            out_mask = self.forward_masked(x, mask, convolve, crops, full_size_image)
        return out_mask, out_raw

    def forward_masked(
        self,
        x: T,
        mask: T,
        convolve: bool = True,
        crops: Optional[T] = None,
        full_size_image: Optional[T] = None,
    ) -> Dict[str, T]:
        """
        Mask the image according to selected masking, then do the forward pass:
        substitute with gaussian noise or local average excluding center pixel (donut)
        :param x:
        :param mask:
        :param convolve: if True, convolve the output with the PSF
        :param crops: if not None, positions of crops `x` in full size image. NB! we assume that crops do not overlap
        :param full_size_image: if not None, full size image from which `x` was taken by `crops` coordinates
        :return: dictionary of outputs:
                    image - final output, always present
                    deconv - output before PSF if PSF is provided and `convolve` is True
        """
        noise = (
            torch.randn(*x.shape, device=x.device, requires_grad=False) * self.noise_std
            + self.noise_mean
            if self.masking == "gaussian"
            else self.mask_kernel(x)
        )

        if isinstance(self.net, SwinIA):
            return self.forward_whole(x, convolve, crops, full_size_image)
        else:
            x = (1 - mask) * x + mask * noise
            return self.forward_whole(x, convolve, crops, full_size_image)

    def forward_whole(
        self,
        x: T,
        convolve: bool = True,
        crops: Optional[T] = None,
        full_size_image: Optional[T] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, T]:
        """
        Plain raw forward pass without masking
        :param x: batch (N, C, H, W, [D])
        :param convolve: if True, convolve the output with the PSF
        :param crops: if not None, positions of crops `x` in full size image. NB! we assume that crops do not overlap
        :param full_size_image: if not None, full size image from which `x` was taken by `crops` coordinates
        :return: dictionary of outputs:
                    image - final output, always present
                    deconv - output before PSF if PSF is provided and `convolve` is True
        """
        out = {}
        features = self.net(x, **kwargs)
        out["image"] = self.head(features)

        if self.residual:
            out["image"] = self.blur(x) + out["image"]

        if self.psf is not None and convolve:
            out["deconv"] = out["image"]

            if crops is not None and full_size_image is not None:
                # convolve with padding approximation from full size image
                for tile, crop in zip(out["image"], crops):
                    # substitute processed crop in blurry image
                    # (we need None for channel axis)
                    image_slice = (slice(None),) + tuple(
                        slice(c, c + ts) for c, ts in zip(crop, x.shape[2:])
                    )
                    full_size_image[image_slice] = tile

                # Convolve full size image with PSF (None emulates batch dimension)
                full_size_image = self.psf(full_size_image[None, ...])[0]
                tiles = []
                for crop in crops:
                    image_slice = (slice(None),) + tuple(
                        slice(c, c + ts) for c, ts in zip(crop, x.shape[2:])
                    )
                    tiles.append(full_size_image[image_slice])
                out["image"] = torch.stack(tiles, dim=0)

            else:
                # just convolve
                out["image"] = self.psf(out["image"])
        return out

    def compute_losses_from_output(
        self,
        x: T,
        mask: T,
        out_mask: Optional[Dict[str, T]] = None,
        out_raw: Optional[Dict[str, T]] = None,
        ground_truth: Optional[T] = None,
    ) -> Tuple[T, Dict[str, float]]:

        loss_log = dict()

        if self.mode in (ModelMode.NOISE2SELF, ModelMode.NOISE2SAME):
            # default Noise2Self blind-spot MSE loss
            bsp_mse = self.compute_mse(x, out_mask["image"], mask=None if isinstance(self.net, SwinIA) else mask)
            loss_log["bsp_mse"] = bsp_mse.item()

        if self.mode == ModelMode.NOISE2SAME:

            # Noise2Same losses
            loss = torch.tensor(0.0, device=x.device)

            # Blind spot MSE
            if self.lambda_bsp > 0:
                loss = loss + self.lambda_bsp * bsp_mse

            # Reconstruction MSE
            rec_mse = self.compute_mse(out_raw["image"], x)
            loss_log["rec_mse"] = rec_mse.item()
            if self.lambda_rec > 0:
                loss = loss + self.lambda_rec * rec_mse

            # Invariance loss for image
            if self.lambda_inv > 0:
                inv_mse = self.compute_mse(
                    out_raw["image"], out_mask["image"], mask=None if isinstance(self.net, SwinIA) else mask
                )
                loss_log["inv_mse"] = inv_mse.item()
                loss = loss + self.lambda_inv * torch.sqrt(inv_mse)

            # Invariance loss for deconv representation
            if self.lambda_inv_deconv > 0 and "deconv" in out_raw:
                inv_deconv_mse = self.compute_mse(
                    out_raw["deconv"], out_mask["deconv"], mask if self.masked_inv_deconv else None,
                )
                loss_log["inv_deconv_mse"] = inv_deconv_mse.item()
                loss = loss + self.lambda_inv_deconv * torch.sqrt(inv_deconv_mse)

        elif self.mode == ModelMode.NOISE2SELF:
            loss = bsp_mse
        elif self.mode == ModelMode.NOISE2TRUE:
            loss = self.compute_mse(out_raw["image"], ground_truth)

        loss_log["loss"] = loss.item()
        return loss, loss_log

    def compute_losses(
        self, x: T, mask: T, convolve: bool = True
    ) -> Tuple[T, Dict[str, float]]:
        out_mask, out_raw = self.forward(x, mask, convolve)
        return self.compute_losses_from_output(x, mask, out_mask, out_raw)

    def compute_regularization_loss(
        self,
        out: Dict[str, T],
        mean: Optional[T] = None,
        std: Optional[T] = None,
        max_value: Optional[T] = None,
    ) -> Tuple[T, Dict[str, float]]:

        x = out[self.regularization_key]

        # todo rewrite in a less ugly way
        if mean is not None and std is not None:
            x = x * std + mean
            if max_value is not None:
                max_value = max_value * std + mean
        if max_value is not None:
            x = x / x.max()

        loss = torch.tensor(0).to(x.device)
        loss_log = {}
        if self.lambda_bound > 0:
            bound_loss = self.lambda_bound * self.compute_boundary_loss(x) ** 2
            loss = bound_loss
            loss_log["bound_loss"] = bound_loss.item()
        if self.lambda_sharp > 0:
            sharp_loss = self.lambda_sharp * self.compute_sharpening_loss(x)
            loss = sharp_loss + loss
            loss_log["sharp_loss"] = sharp_loss.item()

        return loss, loss_log

    @staticmethod
    def compute_boundary_loss(x: T, epsilon: float = 1e-8) -> T:
        bounds_loss = F.relu(-x - epsilon)
        bounds_loss = bounds_loss + F.relu(x - 1 - epsilon)
        return bounds_loss.mean()

    @staticmethod
    def compute_sharpening_loss(x: T) -> T:
        num_elements = x[0, 0].nelement()
        dim = (2, 3) if x.ndim == 4 else (2, 3, 4)
        sharpening_loss = -torch.norm(x, dim=dim, keepdim=True, p=2) / (
            num_elements ** 2
        )
        return sharpening_loss.mean()

    @staticmethod
    def compute_mse(x: T, y: T, mask: Optional[T] = None) -> T:
        if mask is None:
            return torch.mean(torch.square(x - y))
        masked = torch.sum(mask)
        return torch.sum(torch.square(x - y) * mask) / masked
