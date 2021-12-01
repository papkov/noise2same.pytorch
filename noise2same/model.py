from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor as T
from torch import nn
from torch.nn.functional import conv2d, conv3d
from torchvision.transforms import GaussianBlur

from noise2same import network
from noise2same.contrast import PixelContrastLoss
from noise2same.psf import PSF, PSFParameter, read_psf


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
        lambda_inv: float = 2.0,
        mask_percentage: float = 0.5,
        masking: str = "gaussian",
        noise_mean: float = 0,
        noise_std: float = 0.2,
        lambda_proj: float = 0,
        psf: Optional[str] = None,
        psf_size: Optional[int] = None,
        psf_pad_mode: str = "reflect",
        residual: bool = False,
        skip_method: str = "concat",
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
        :param lambda_proj:
        :param psf:
        """
        super(Noise2Same, self).__init__()
        assert masking in ("gaussian", "donut")
        self.n_dim = n_dim
        self.in_channels = in_channels
        self.lambda_inv = lambda_inv
        self.lambda_proj = lambda_proj
        self.mask_percentage = mask_percentage
        self.masking = masking
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.residual = residual

        # TODO customize with segmentation_models
        self.net = network.UNet(
            in_channels=in_channels,
            n_dim=n_dim,
            base_channels=base_channels,
            skip_method=skip_method,
            **kwargs,
        )
        self.head = network.RegressionHead(
            in_channels=base_channels,
            out_channels=in_channels,
            n_dim=n_dim,
        )

        # todo parametrize
        self.blur = GaussianBlur(5, sigma=0.2) if residual else None

        # TODO parametrize project head
        self.project_head = None
        if self.lambda_proj > 0:
            self.project_head = network.ProjectHead(
                in_channels=base_channels, n_dim=n_dim, out_channels=256, kernel_size=1
            )

        self.mask_kernel = DonutMask(n_dim=n_dim, in_channels=in_channels)

        self.psf = None
        if psf is not None:
            psf = read_psf(psf, psf_size=psf_size)
            print("PSF shape", psf.shape)
            self.psf = PSFParameter(psf, pad_mode=psf_pad_mode)
            for param in self.psf.parameters():
                param.requires_grad = False

    def forward_full(self, x: T, mask: T) -> Tuple[Dict[str, T], Dict[str, T]]:
        """
        Make two forward passes: with mask and without mask
        :param x:
        :param mask:
        :return: tuple of tensors: output for masked input, output for raw input
        """
        out_mask = self.forward_masked(x, mask)
        out_raw = self.forward(x)
        return out_mask, out_raw

    def forward_masked(self, x: T, mask: T) -> Dict[str, T]:
        """
        Mask the image according to selected masking, then do the forward pass:
        substitute with gaussian noise or local average excluding center pixel (donut)
        :param x:
        :param mask:
        :return:
        """
        noise = (
            torch.randn(*x.shape, device=x.device, requires_grad=False) * self.noise_std
            + self.noise_mean
            # np.random.normal(self.noise_mean, self.noise_std, x.shape)
            if self.masking == "gaussian"
            else self.mask_kernel(x)
        )
        x = (1 - mask) * x + mask * noise
        return self.forward(x)

    def forward(self, x: T, *args: Any, **kwargs: Any) -> Dict[str, T]:
        """
        Plain raw forward pass without masking
        :param x:
        :return:
        """
        out = {}
        features = self.net(x)
        out["image"] = self.head(features)

        if self.residual:
            out["image"] = self.blur(x) + out["image"]

        if self.psf is not None:
            out["deconv"] = out["image"]
            out["image"] = self.psf(out["image"])
        if self.project_head is not None:
            out["proj"] = self.project_head(features)
        return out

    def compute_losses_from_output(
        self, x: T, mask: T, out_mask: Dict[str, T], out_raw: Dict[str, T]
    ) -> Tuple[T, Dict[str, float]]:
        masked = torch.sum(mask)
        rec_mse = torch.mean(torch.square(out_raw["image"] - x))
        inv_mse = (
            torch.sum(torch.square(out_raw["image"] - out_mask["image"]) * mask)
            / masked
        )
        bsp_mse = torch.sum(torch.square(x - out_mask["image"]) * mask) / masked
        loss = rec_mse + self.lambda_inv * torch.sqrt(inv_mse)
        loss_log = {
            "loss": loss.item(),
            "rec_mse": rec_mse.item(),
            "inv_mse": inv_mse.item(),
            "bsp_mse": bsp_mse.item(),
        }
        if self.lambda_proj > 0:
            contrastive_loss = PixelContrastLoss(temperature=0.1)
            proj_loss = contrastive_loss(out_raw["proj"], out_mask["proj"], mask).mean()
            loss = loss + self.lambda_proj * proj_loss
            loss_log.update({"loss": loss.item(), "proj_loss": proj_loss.item()})

        return loss, loss_log

    def compute_losses(self, x: T, mask: T) -> Tuple[T, Dict[str, float]]:
        out_mask, out_raw = self.forward_full(x, mask)
        return self.compute_losses_from_output(x, mask, out_mask, out_raw)
