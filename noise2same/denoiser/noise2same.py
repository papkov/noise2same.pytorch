from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor as T

from noise2same.denoiser import Noise2Self


class Noise2Same(Noise2Self):
    """
    Noise2Same denoiser implementation.
    It makes masked and unmasked forward passes and calculates an additional
    invariance loss between them.
    """
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_inv: float = 2.0,
        lambda_bsp: float = 0.0,
        **kwargs: Any,
    ):
        """
        Noise2Same denoiser implementation.
        :param lambda_rec: reconstruction loss weight (input vs output)
        :param lambda_inv: invariance loss weight  (output vs masked output)
        :param lambda_bsp: blind-spot loss weight (input vs masked output)
        """
        super().__init__(**kwargs)
        self.lambda_rec = lambda_rec
        self.lambda_inv = lambda_inv
        self.lambda_bsp = lambda_bsp

    def forward(self, x: T, mask: Optional[T] = None) -> Dict[str, T]:
        out = super().forward(x)
        if mask is not None:
            out["image/masked"] = super().forward(x, mask)["image"]
        return out

    def compute_loss(self, x_in: Dict[str, T], x_out: Dict[str, T]) -> Tuple[T, Dict[str, float]]:
        bsp_mse = self.compute_mse(x_in['image'], x_out['image/masked'], mask=x_in['mask'])
        inv_mse = self.compute_mse(x_out['image'], x_out['image/masked'])
        rec_mse = self.compute_mse(x_in['image'], x_out['image'])

        loss = self.lambda_bsp * bsp_mse + self.lambda_inv * torch.sqrt(inv_mse) + self.lambda_rec * rec_mse
        loss_dict = {'loss': loss.item(),
                     'bsp_mse': bsp_mse.item(),
                     'inv_mse': inv_mse.item(),
                     'rec_mse': rec_mse.item()}

        return loss, loss_dict