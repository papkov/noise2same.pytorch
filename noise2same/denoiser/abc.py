from typing import Optional, Dict, Tuple

import torch
from torch import Tensor as T
from torch import nn


class Denoiser(nn.Module):
    """
    Base class for denoising networks. Implements autoencoder-like architecture
    with backbone and head. Computes MSE loss between input and output images or
    between ground truth and output images if ground truth is available.
    """

    def __init__(
        self,
        backbone: nn.Module = nn.Identity(),
        head: nn.Module = nn.Identity(),
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.backbone = backbone
        self.head = head

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
        :param x_in: input dictionary, must contain 'image' key, optionally 'gt' and 'mask' keys
        :param x_out: model output dictionary, must contain 'image' key
        :return: loss tensor for backpropagation and dictionary with loss values for logging
        """
        loss = self.compute_mse(x_in['gt' if 'gt' in x_in else 'image'], x_out['image'])
        loss_dict = {'loss': loss.item()}
        return loss, loss_dict

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
