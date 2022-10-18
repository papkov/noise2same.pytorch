from .decoder import UPerHead
from .encoder import ConvSwin
from .encoder_decoder import EncoderDecoderDenoiser


class ConvSwinUPer(EncoderDecoderDenoiser):

    def __init__(self,
                 img_size: int,
                 in_chans: int,
                 embed_dim: int,
                 uper_channels: int,
                 n_dim: int = 2,
                 align_corners: bool = False,
                 **kwargs
                 ):
        swin = ConvSwin(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            n_dim=n_dim,
            **kwargs
        )
        uper = UPerHead(
            num_classes=in_chans,
            channels=uper_channels,
            in_channels=[embed_dim, *[embed_dim * 2 ** i for i in range(4)]],
            in_index=list(range(5)),
            align_corners=align_corners
        )
        super(ConvSwinUPer, self).__init__(swin, uper, rescale=False, align_corners=align_corners)
