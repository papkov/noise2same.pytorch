import torch
from torch import nn
from torch import Tensor as T
from typing import Optional, Dict

from network.backbone import AbstractBackbone
from network.head import AbstractHead


class Denoiser(nn.Module):

    def __init__(
        self,
        backbone: AbstractBackbone = nn.Identity(),
        head: AbstractHead = nn.Identity()
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: T):
        return self.head(self.backbone(x))

    def compute_loss(self, x_in: Dict[str, T], x_out: Dict[str, T]):
        if 'gt' in x_in:
            return self.compute_mse(x_in['gt'], x_out['image'])
        return self.compute_mse(x_in['image'], x_out['image'])

    @staticmethod
    def compute_mse(x: T, y: T, mask: Optional[T] = None) -> T:
        if mask is None:
            return torch.mean(torch.square(x - y))
        masked = torch.sum(mask)
        return torch.sum(torch.square(x - y) * mask) / masked
