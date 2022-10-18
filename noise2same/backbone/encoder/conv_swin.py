from .swin import SwinTransformer
from ..ops import ConvBlock
import torch.nn as nn


class ConvSwin(nn.Module):

    def __init__(self,
                 img_size: int,
                 in_chans: int,
                 embed_dim: int,
                 n_dim: int,
                 **kwargs):
        super(ConvSwin, self).__init__()
        self.conv = ConvBlock(in_chans, embed_dim, 3, padding=1, n_dim=n_dim)
        self.swin = SwinTransformer(
            pretrain_img_size=img_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            **kwargs
        )

    def forward(self, x):
        first_out = self.conv(x)
        swin_outs = self.swin(first_out)
        return first_out, *swin_outs
