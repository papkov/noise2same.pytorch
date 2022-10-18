import torch.nn as nn
from torch import Tensor as T

from noise2same.backbone.ops import resize


class EncoderDecoderDenoiser(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 rescale: bool = False,
                 align_corners: bool = False):
        super(EncoderDecoderDenoiser, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rescale = rescale
        self.align_corners = align_corners

    def forward(self, x: T):
        features = self.encoder(x)
        output = self.decoder(features)
        if self.rescale:
            output = resize(
                output,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return output

