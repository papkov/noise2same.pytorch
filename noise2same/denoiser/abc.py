from typing import Any, Optional, Dict, Tuple

import torch
from torch import Tensor as T
from torch import nn
from torch.nn import functional as F


class Denoiser(nn.Module):
    """
    Base class for denoising networks. Implements autoencoder-like architecture
    with backbone and head. Computes MSE loss between input and output images or
    between ground truth and output images if ground truth is available.
    """

    def __init__(
            self,
            # TODO consider removing default values
            backbone: nn.Module = nn.Identity(),
            head: nn.Module = nn.Identity(),
            residual: bool = False,
            target_key: str = "image",
    ):
        super().__init__()
        self.residual = residual
        self.backbone = backbone
        self.head = head
        self.target_key = target_key

    def forward(self, x: T, mask: Optional[T] = None) -> Dict[str, T]:
        """
        Forward pass through the network's backbone and head
        :param x: input tensor
        :param mask: optional binary mask tensor
        :return:
        """
        out = self.head(self.backbone(x))
        if self.residual:
            out += x
        return {'image': out}

    def compute_loss(self, x_in: Dict[str, T], x_out: Dict[str, T]) -> Tuple[T, Dict[str, float]]:
        """
        Computes MSE loss between input and output images or between ground truth
        :param x_in: input dictionary, must contain 'image' key, optionally 'ground_truth' and 'mask' keys
        :param x_out: model output dictionary, must contain 'image' key
        :return: loss tensor for backpropagation and dictionary with loss values for logging
        """
        loss = self.compute_mse(x_in[self.target_key], x_out['image'])
        return loss, {'loss': loss.item(), 'rec_mse': loss.item()}

    @staticmethod
    def compute_mse(x: T, y: T, mask: Optional[T] = None) -> T:
        """
        Computes MSE loss between x and y, optionally in masked regions
        :param x: input tensor
        :param y: reference tensor
        :param mask: binary mask tensor
        :return: MSE loss
        """
        if mask is None:
            return torch.mean(torch.square(x - y))
        masked = torch.sum(mask)
        return torch.sum(torch.square(x - y) * mask) / masked


class DeconvolutionMixin:
    """
    Mixin for deconvolutional networks.
    Convolves outputs of the backbone with a fixed PSF kernel.
    """
    def __init__(
        self,
        psf: nn.Module,
        lambda_bound: float = 0.0,
        lambda_sharp: float = 0.0,
        regularization_key: str = "image",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.psf = psf
        self.lambda_bound = lambda_bound
        self.lambda_sharp = lambda_sharp
        self.regularization_key = regularization_key

    def forward(self, x: T, mask: Optional[T] = None) -> Dict[str, T]:
        """
        Pass all network's outputs through the PSF convolution
        :param x: input tensor
        :param mask: binary mask tensor
        :return: dict of outputs before and after convolution (with and without '/deconv' suffix')
        """
        out = super().forward(x, mask)
        out_convolved = {k: self.psf(v) for k, v in out.items()}
        out = {f'{k}/deconv': v for k, v in out.items()}
        out.update(out_convolved)
        return out

    def inference(self, x: T) -> Dict[str, T]:
        return super().forward(x)

    def compute_regularization_loss(
            self,
            x_in: Dict[str, T],
            x_out: Dict[str, T],
    ) -> Tuple[T, Dict[str, float]]:

        x = x_out[self.regularization_key] * x_in['std'] + x_in['mean']
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
        n_elements_sq = x[0, 0].nelement() ** 2
        dim = (2, 3) if x.ndim == 4 else (2, 3, 4)
        sharpening_loss = -torch.norm(x, dim=dim, keepdim=True, p=2) / n_elements_sq
        return sharpening_loss.mean()
