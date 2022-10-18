from noise2same.backbone.encoder_decoder import EncoderDecoderDenoiser
from noise2same.backbone.encoder.swin import SwinTransformer
from noise2same.backbone.decoder.uper_head import UPerHead


class SwinUPer(EncoderDecoderDenoiser):

    def __init__(self,
                 img_size: int,
                 in_chans: int,
                 embed_dim: int,
                 uper_channels: int,
                 align_corners: bool = False,
                 **kwargs
                 ):
        swin = SwinTransformer(
            pretrain_img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            **kwargs
        )
        uper = UPerHead(
            num_classes=in_chans,
            channels=uper_channels,
            in_channels=[embed_dim * 2 ** i for i in range(4)],
            in_index=list(range(4)),
            align_corners=align_corners
        )
        super(SwinUPer, self).__init__(swin, uper, rescale=True, align_corners=align_corners)
