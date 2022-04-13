# translated from
# https://github.com/divelab/Noise2Same/blob/main/network.py
# https://github.com/divelab/Noise2Same/blob/main/resnet_module.py
from functools import partial
from typing import Tuple, Any

import torch
from torch import Tensor as T
from torch import nn
from torch.nn.functional import normalize

from noise2same.ffc import BN_ACT_FFC, FFC


class ProjectHead(nn.Sequential):
    """
    Implements projection head for contrastive learning as per
    "Exploring Cross-Image Pixel Contrast for Semantic Segmentation"
    https://arxiv.org/abs/2101.11939
    https://github.com/tfzhou/ContrastiveSeg

    Provides high-dimensional L2-normalized pixel embeddings (256-d from 1x1 conv by default)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int = 256,
            n_dim: int = 2,
            kernel_size: int = 1,
    ):
        assert n_dim in (2, 3)
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        conv_1 = conv(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        relu = nn.ReLU(inplace=True)
        conv_2 = conv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        super().__init__(conv_1, relu, conv_2, relu)

    def forward(self, x):
        x = super().forward(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class RegressionHead(nn.Sequential):
    def __init__(
            self, in_channels: int, out_channels: int, n_dim: int = 2, kernel_size: int = 1
    ):
        """
        Denoising regression head BN-ReLU-Conv

        https://github.com/divelab/Noise2Same/blob/main/models.py
        :param in_channels:
        :param out_channels:
        :param n_dim:
        :param kernel_size:
        """
        assert n_dim in (2, 3)
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d

        bn = bn(num_features=in_channels)
        relu = nn.ReLU(inplace=True)
        conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        super().__init__(bn, relu, conv)


class ResidualUnit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_dim: int = 2,
            kernel_size: int = 3,
            downsample: bool = False,
            ffc: bool = False,
            **kwargs: Any
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.ffc = ffc

        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        stride = 2 if downsample else 1

        self.act = nn.ReLU(inplace=True)
        # todo parametrize as in the original repo (bn momentum is inverse)

        bn_in_channels = in_channels
        conv_shortcut = conv
        if ffc:
            bn_in_channels = bn_in_channels // 2
            conv_shortcut = partial(
                BN_ACT_FFC, n_dim=n_dim, ratio_gin=0.5, ratio_gout=0, bn_act_first=True
            )

        self.conv_shortcut = conv_shortcut(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )

        self.bn = bn(bn_in_channels, momentum=1 - 0.997, eps=1e-5)

        if self.ffc:

            bnactffc = partial(
                BN_ACT_FFC,
                stride=1,
                activation_layer=nn.ReLU,
                enable_lfu=True,
                kernel_size=3,
                padding=1,
                n_dim=n_dim,
                bn_act_first=True,
            )
            self.layers = nn.Sequential(
                bnactffc(
                    ratio_gin=0.5,
                    ratio_gout=0.5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                ),
                bnactffc(
                    ratio_gin=0.5,
                    ratio_gout=0,
                    in_channels=out_channels,
                    out_channels=out_channels,
                ),
            )
        else:
            self.layers = nn.Sequential(
                conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 if downsample else kernel_size,
                    padding=0 if downsample else kernel_size // 2,
                    stride=stride,
                    bias=False,
                ),
                bn(out_channels),
                self.act,
                conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1,
                    bias=False,
                ),
            )

    def forward(self, x: T) -> T:

        if self.ffc:
            shortcut = self.conv_shortcut(x)[0]
        else:
            shortcut = x
            x = self.bn(x)
            x = self.act(x)
            if self.in_channels != self.out_channels or self.downsample:
                shortcut = self.conv_shortcut(x)

        x = self.layers(x)
        if type(x) == tuple:
            x = x[0]
        return x + shortcut


class ResidualUnitExtraLayer(ResidualUnit):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_dim: int = 2,
            kernel_size: int = 3,
            downsample: bool = False,
            ffc: bool = False
    ):
        super(ResidualUnitExtraLayer, self).__init__(in_channels=in_channels, out_channels=out_channels, n_dim=n_dim,
                                                     ffc=ffc)

        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        stride = 2 if downsample else 1
        conv_shortcut = conv

        self.conv_shortcut = conv_shortcut(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )
        if self.ffc:

            bnactffc = partial(
                BN_ACT_FFC,
                stride=1,
                activation_layer=nn.ReLU,
                enable_lfu=True,
                kernel_size=3,
                padding=1,
                n_dim=n_dim,
                bn_act_first=True)
            self.layers = nn.Sequential(
                bnactffc(
                    ratio_gin=0,
                    ratio_gout=0.5,
                    in_channels=in_channels,
                    out_channels=out_channels,
                ),
                bnactffc(
                    ratio_gin=0.5,
                    ratio_gout=0.5,
                    in_channels=out_channels,
                    out_channels=out_channels,
                ),
                bnactffc(
                    ratio_gin=0.5,
                    ratio_gout=0,
                    in_channels=out_channels,
                    out_channels=out_channels,
                )
            )
        else:
            self.layers = nn.Sequential(
                bn(in_channels),
                self.act,
                conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 if downsample else kernel_size,
                    padding=0 if downsample else kernel_size // 2,
                    stride=stride,
                    bias=False,
                ),
                bn(out_channels),
                self.act,
                conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1,
                    bias=False,
                ),
                bn(out_channels),
                self.act,
                conv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1,
                    bias=False,
                ),
            )

    def forward(self, x: T) -> T:

        shortcut = self.conv_shortcut(x)
        x = self.layers(x)
        return x[0] + shortcut


class ResidualUnitDualPass(ResidualUnit):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_dim: int = 2,
            kernel_size: int = 3,
            downsample: bool = False,
            ffc: bool = False,
            last_block: bool = False
    ):
        super(ResidualUnitDualPass, self).__init__(in_channels=in_channels, out_channels=out_channels, n_dim=n_dim,
                                                   ffc=ffc)

        self.last_block = last_block

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        stride = 2 if downsample else 1
        conv_shortcut = conv

        self.conv_shortcut_local = conv_shortcut(
            in_channels=in_channels // 2,
            out_channels=out_channels // (1 if last_block else 2),
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )
        self.conv_shortcut_global = conv_shortcut(
            in_channels=in_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )
        if self.ffc:

            bnactffc = partial(
                BN_ACT_FFC,
                ratio_gin=0.5,
                ratio_gout=0.5,
                stride=1,
                activation_layer=nn.ReLU,
                enable_lfu=True,
                kernel_size=3,
                padding=1,
                n_dim=n_dim,
                bn_act_first=True,
            )
            self.layers = nn.Sequential(
                bnactffc(
                    in_channels=in_channels,
                    out_channels=out_channels,
                ),
                bnactffc(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    ratio_gout=0 if last_block else 0.5
                ),
            )
        else:
            raise ValueError("Dual pass possible only with ffc=True")

    def forward(self, x: T) -> T:

        if self.ffc:
            x_l, x_g = x if type(x) is tuple else (x, 0)
            shortcut_local = self.conv_shortcut_local(x_l)
            shortcut_global = self.conv_shortcut_global(x_g)

            x = self.layers(x)

            x_l, x_g = x if type(x) is tuple else (x, 0)
            x_l = x_l + shortcut_local
            x_g = x_g + shortcut_global
            x = (x_l, x_g)

            if self.last_block:
                x = x_l
        else:
            shortcut = self.conv_shortcut_local(x)
            x = self.layers(x)
            x = x + shortcut

        return x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_size: int = 1,
            n_dim: int = 2,
            kernel_size: int = 3,
            downsample: bool = False,
            ffc: bool = False,
            unit_type: str = 'default',
            last_block: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.block_size = block_size

        if unit_type == 'default':
            unit = ResidualUnit
        elif unit_type == 'extra_layer':
            unit = ResidualUnitExtraLayer
        elif unit_type == 'dual_pass':
            unit = ResidualUnitDualPass
        else:
            raise ValueError("unit_type", unit_type, "not supported")

        self.block = nn.Sequential(
            *[
                unit(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    ffc=ffc,
                    downsample=downsample if i == 0 else False,
                    last_block=last_block
                )
                for i in range(0, block_size)
            ]
        )

    def forward(self, x: T) -> T:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_size: int = 1,
            n_dim: int = 2,
            kernel_size: int = 3,
            downsampling: str = "conv",
            ffc: bool = False,
            unit_type: str = "default"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.unit_type = unit_type

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        if unit_type == "default" and ffc:
            conv = partial(FFC, n_dim=n_dim, ratio_gin=0, ratio_gout=0.5)

        if downsampling == "res":
            self.downsampling_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                n_dim=n_dim,
                kernel_size=kernel_size,
                block_size=1,
                downsample=True,
            )
        elif downsampling == "conv":
            downsample = partial(conv, kernel_size=2, stride=2, bias=True)
            if unit_type == "dual_pass":
                self.downsampling_block_local = downsample(in_channels=in_channels // 2, out_channels=out_channels // 2)
                self.downsampling_block_global = downsample(in_channels=in_channels // 2,
                                                            out_channels=out_channels // 2)
            else:
                self.downsampling_block = downsample(in_channels=in_channels, out_channels=out_channels)
        else:
            raise ValueError("downsampling should be `res`. `conv`, `pool`")

        self.resblock = ResidualBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            n_dim=n_dim,
            block_size=block_size,
            downsample=False,
            kernel_size=kernel_size,
            ffc=ffc,
            unit_type=unit_type)

    def forward(self, x: T) -> T:
        if self.unit_type == "dual_pass":
            x_l, x_g = x
            x_l = self.downsampling_block_local(x_l)
            x_g = self.downsampling_block_global(x_g)
            x = (x_l, x_g)
        else:
            x = self.downsampling_block(x)
        x = self.resblock(x)
        return x


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            base_channels: int = 96,
            kernel_size: int = 3,
            n_dim: int = 2,
            depth: int = 3,
            encoding_block_sizes: Tuple[int, ...] = (1, 1, 0),
            decoding_block_sizes: Tuple[int, ...] = (1, 1),
            downsampling: Tuple[str, ...] = ("conv", "conv"),
            upsampling: Tuple[str, ...] = ("conv", "conv"),
            skip_method: str = "concat",
            ffc: bool = False,
            unit_type: str = 'default'
    ):
        """

        configuration: https://github.com/divelab/Noise2Same/blob/main/network_configure.py
        architecture: https://github.com/divelab/Noise2Same/blob/main/network.py

        :param n_dim:
        :param depth:
        :param base_channels:
        :param encoding_block_sizes:
        :param decoding_block_sizes:
        :param downsampling:
        :param upsampling:
        :param skip_method:
        :param ffc:
        """
        super().__init__()

        assert depth == len(encoding_block_sizes)
        assert encoding_block_sizes[0] > 0
        assert encoding_block_sizes[-1] == 0
        assert depth == len(decoding_block_sizes) + 1
        assert depth == len(downsampling) + 1
        assert len(downsampling) == len(upsampling)
        assert skip_method in ["add", "concat", "cat"]

        self.in_channels = in_channels
        self.n_dim = n_dim
        self.depth = depth
        self.base_channels = base_channels
        self.encoding_block_sizes = encoding_block_sizes
        self.decoding_block_sizes = decoding_block_sizes
        self.downsampling = downsampling
        self.skip_method = skip_method
        self.ffc = ffc
        self.unit_type = unit_type
        print(f"Use {self.skip_method} skip method")

        if ffc and unit_type != "extra_layer":
            conv = partial(FFC, n_dim=n_dim, ratio_gin=0, ratio_gout=0.5)
        else:
            conv = nn.Conv2d if n_dim == 2 else nn.Conv3d

        conv_transpose = nn.ConvTranspose2d if n_dim == 2 else nn.ConvTranspose3d

        self.conv_first = conv(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
            bias=False,
        )

        # reset from FFC for now -- may need to change for the decoder!
        # conv = nn.Conv2d if n_dim == 2 else nn.Conv3d

        # Encoder
        self.encoder_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=encoding_block_sizes[0],
                    ffc=ffc,
                    unit_type=unit_type
                )
            ]
        )

        out_channels = base_channels
        for i in range(2, self.depth + 1):
            in_channels = base_channels * (2 ** (i - 2))
            out_channels = base_channels * (2 ** (i - 1))

            # todo downsampling
            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=encoding_block_sizes[i - 1],
                    downsampling=downsampling[i - 2],
                    ffc=ffc,
                    unit_type=unit_type
                )
            )

        # Bottom block
        self.bottom_block = ResidualBlock(
            in_channels=out_channels,
            out_channels=base_channels * (2 ** (depth - 1)),
            n_dim=n_dim,
            kernel_size=kernel_size,
            block_size=1,
            ffc=ffc,
            unit_type=unit_type
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsampling_blocks_main = nn.ModuleList()
        self.upsampling_blocks_secondary = nn.ModuleList()

        for i in range(self.depth - 1, 0, -1):

            in_channels = int(base_channels * (2 ** i))
            out_channels = int(base_channels * (2 ** (i - 1)))

            if upsampling[i - 1] == "conv":
                upsample = partial(conv_transpose, kernel_size=2,
                                   stride=2,
                                   bias=True)
                if unit_type == "dual_pass":
                    upsampling_block_main = upsample(in_channels=in_channels // 2,
                                                     out_channels=out_channels // 2)  # local
                    upsampling_block_secondary = upsample(in_channels=in_channels // 2,
                                                          out_channels=out_channels // 2)  # global
                else:
                    upsampling_block_main = upsample(in_channels=in_channels, out_channels=out_channels)
                    upsampling_block_secondary = nn.Identity()

            else:
                raise ValueError(
                    f"Upsampling method {upsampling[i - 1]} not supported for {n_dim}D")

            self.upsampling_blocks_main.append(upsampling_block_main)
            self.upsampling_blocks_secondary.append(upsampling_block_secondary)

            # Here goes skip connection, then decoder block
            self.decoder_blocks.append(
                ResidualBlock(
                    in_channels=out_channels * (2 if self.skip_method != "add" else 1),  # *2
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=decoding_block_sizes[depth - 1 - i],
                    ffc=ffc if unit_type != "default" else False,
                    unit_type=unit_type,
                    last_block=True if i == 1 else False
                ))

    def forward(self, x: T) -> T:
        encoder_outputs = []
        x = self.conv_first(x)
        x = self.encoder_blocks[0](x)

        for i, encoder_block in enumerate(self.encoder_blocks[1:]):
            encoder_outputs.append(x)
            x = encoder_block(x)

        x = self.bottom_block(x)

        for i, (upsampling_block_local, upsampling_block_global, decoder_block, skip) in enumerate(
                zip(
                    self.upsampling_blocks_main,
                    self.upsampling_blocks_secondary,
                    self.decoder_blocks,
                    encoder_outputs[::-1],
                )
        ):

            if self.unit_type == "dual_pass":
                x_l, x_g = x
                x_l = upsampling_block_local(x_l)
                x_g = upsampling_block_global(x_g)

                if self.skip_method == "add":
                    x_l.add_(skip[0])
                    x_g.add_(skip[1])
                elif self.skip_method in ("cat", "concat"):
                    x_l = torch.cat([x_l, skip[0]], dim=1)
                    x_g = torch.cat([x_g, skip[1]], dim=1)
                    x = (x_l, x_g)
            else:
                x = upsampling_block_local(x)
                if self.skip_method == "add":
                    x.add_(skip)
                elif self.skip_method in ("cat", "concat"):
                    x = torch.cat([x, skip], dim=1)

            x = decoder_block(x)

        return x
