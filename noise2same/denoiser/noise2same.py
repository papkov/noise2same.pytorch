from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor as T

from noise2same.denoiser import Noise2Self
from noise2same.denoiser.abc import DeconvolutionMixin


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
        inv_mse = self.compute_mse(x_out['image'], x_out['image/masked'], mask=x_in['mask'])
        rec_mse = self.compute_mse(x_in['image'], x_out['image'])

        loss = self.lambda_bsp * bsp_mse + self.lambda_inv * torch.sqrt(inv_mse) + self.lambda_rec * rec_mse
        loss_dict = {'loss': loss.item(),
                     'bsp_mse': bsp_mse.item(),
                     'inv_mse': inv_mse.item(),
                     'rec_mse': rec_mse.item()}

        return loss, loss_dict


class Noise2SameDeconvolution(DeconvolutionMixin, Noise2Same):
    def __init__(self, lambda_inv_deconv: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.lambda_inv_deconv = lambda_inv_deconv

    def compute_loss(self, x_in: Dict[str, T], x_out: Dict[str, T]) -> Tuple[T, Dict[str, float]]:
        """
        Computes loss for Noise2SameDeconvolution.
        In addition to Noise2Same loss, it also calculates invariance loss for deconvolved representations before PSF
        and regularization losses.
        :param x_in: input tensor dict, should contain 'image', 'mask', 'mean' and 'std' keys
        :param x_out: output tensor dict, should contain 'image', 'image/deconv',
                      'image/masked' and 'image/masked/deconv' keys
        :return:
        """
        loss, loss_dict = super().compute_loss(x_in, x_out)
        # TODO test deconvolved invariance loss without mask
        inv_deconv_mse = self.compute_mse(x_out['image/deconv'], x_out['image/masked/deconv'], mask=x_in['mask'])
        loss_dict['inv_deconv_mse'] = inv_deconv_mse.item()
        loss = loss + self.lambda_inv_deconv * torch.sqrt(inv_deconv_mse)

        regularization_loss, regularization_loss_dict = super().compute_regularization_loss(x_in, x_out)
        loss = loss + regularization_loss
        loss_dict.update(regularization_loss_dict)
        loss_dict['loss'] = loss.item()
        return loss, loss_dict
