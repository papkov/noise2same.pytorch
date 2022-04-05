# translated from
# https://github.com/divelab/Noise2Same/blob/main/network.py
# https://github.com/divelab/Noise2Same/blob/main/resnet_module.py
from functools import partial
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from torch.nn.functional import normalize

from noise2same.ffc import BN_ACT_FFC, FFC, FFCInc


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
        ratio_ffc: float = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.ffc = ratio_ffc > 0

        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        stride = 2 if downsample else 1

        self.act = nn.ReLU(inplace=True)
        # todo parametrize as in the original repo (bn momentum is inverse)

        bn_in_channels = in_channels
        conv_shortcut = conv
        if self.ffc:
            bn_in_channels = bn_in_channels // 2
            conv_shortcut = partial(
                BN_ACT_FFC,
                n_dim=n_dim,
                ratio_gin=0.5,
                ratio_gout=0,
                ratio_ffc=ratio_ffc,
                bn_act_first=True,
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
            ffc_params = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                activation_layer=nn.ReLU,  # TODO this should be set separately for each layer
                enable_lfu=True,
                kernel_size=3,
                padding=1,
                n_dim=n_dim,
                bn_act_first=True,
                ratio_ffc=ratio_ffc,
            )
            self.layers = nn.Sequential(
                BN_ACT_FFC(**ffc_params, ratio_gin=0.5, ratio_gout=0.5),
                BN_ACT_FFC(**ffc_params, ratio_gin=0.5, ratio_gout=0),
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


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_size: int = 1,
        n_dim: int = 2,
        kernel_size: int = 3,
        downsample: bool = False,
        ratio_ffc: float = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.block_size = block_size

        self.block = nn.Sequential(
            *[
                ResidualUnit(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    ratio_ffc=ratio_ffc,
                    downsample=downsample if i == 0 else False,
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
        ratio_ffc: float = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.ffc = ratio_ffc > 0

        if self.ffc:
            conv = partial(
                FFC, n_dim=n_dim, ratio_gin=0, ratio_gout=0.5, ratio_ffc=ratio_ffc
            )
        else:
            conv = nn.Conv2d if n_dim == 2 else nn.Conv3d

        if downsampling == "res":
            downsampling_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                n_dim=n_dim,
                kernel_size=kernel_size,
                block_size=1,
                downsample=True,
                # todo ffc was not parametrized here. for a reason?
                # ratio_ffc=ratio_ffc,
            )
        elif downsampling == "conv":
            downsampling_block = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                bias=True,
                # todo ffc was not parametrized here. for a reason?
                # ratio_ffc=ratio_ffc,
            )
        else:
            raise ValueError("downsampling should be `res`. `conv`, `pool`")

        self.block = nn.Sequential(
            downsampling_block,
            ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_dim=n_dim,
                block_size=block_size,
                downsample=False,
                kernel_size=kernel_size,
                ratio_ffc=ratio_ffc,
            ),
        )

    def forward(self, x: T) -> T:
        x = self.block(x)
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
        skip_method: str = "concat",
        ratio_ffc: float = 0,
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
        :param skip_method:
        :param ratio_ffc:
        """
        super().__init__()

        assert depth == len(encoding_block_sizes)
        assert encoding_block_sizes[0] > 0
        assert encoding_block_sizes[-1] == 0
        assert depth == len(decoding_block_sizes) + 1
        assert depth == len(downsampling) + 1
        assert skip_method in ["add", "concat", "cat"]

        self.in_channels = in_channels
        self.n_dim = n_dim
        self.depth = depth
        self.base_channels = base_channels
        self.encoding_block_sizes = encoding_block_sizes
        self.decoding_block_sizes = decoding_block_sizes
        self.downsampling = downsampling
        self.skip_method = skip_method
        self.ffc = ratio_ffc > 0
        print(f"Use {self.skip_method} skip method")

        if self.ffc:
            conv = partial(
                FFCInc, n_dim=n_dim, ratio_gin=0, ratio_gout=0.5, ratio_ffc=ratio_ffc
            )
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

        # Encoder
        self.encoder_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=encoding_block_sizes[0],
                    ratio_ffc=ratio_ffc,
                )
            ]
        )

        out_channels = base_channels
        for i in range(2, self.depth + 1):
            in_channels = base_channels * (2 ** (i - 2))
            out_channels = base_channels * (2 ** (i - 1))

            # Here

            # todo downsampling

            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=encoding_block_sizes[i - 1],
                    downsampling=downsampling[i - 2],
                    ratio_ffc=ratio_ffc,
                )
            )

        # Bottom block
        self.bottom_block = ResidualBlock(
            in_channels=out_channels,
            out_channels=base_channels * (2 ** (depth - 1)),
            n_dim=n_dim,
            kernel_size=kernel_size,
            block_size=1,
            ratio_ffc=ratio_ffc,
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            in_channels = int(base_channels * (2 ** i))
            out_channels = int(base_channels * (2 ** (i - 1)))

            # todo parametrize to use linear upsampling optionally
            self.upsampling_blocks.append(
                conv_transpose(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    bias=True,
                )
            )

            # Here goes skip connection, then decoder block
            self.decoder_blocks.append(
                ResidualBlock(
                    in_channels=out_channels * (2 if self.skip_method != "add" else 1),
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=decoding_block_sizes[depth - 1 - i],
                )
            )

    def forward(self, x: T) -> T:
        encoder_outputs = []
        x = self.conv_first(x)
        # print("First conv", x.shape)
        x = self.encoder_blocks[0](x)
        # print("Encoder 0", x.shape)

        for i, encoder_block in enumerate(self.encoder_blocks[1:]):
            encoder_outputs.append(x)
            x = encoder_block(x)
            # print(f"Encoder {i+1}", x.shape)
        x = self.bottom_block(x)

        for i, (upsampling_block, decoder_block, skip) in enumerate(
            zip(self.upsampling_blocks, self.decoder_blocks, encoder_outputs[::-1])
        ):
            x = upsampling_block(x)
            # print(f"Upsampling {i}", x.shape)
            if self.skip_method == "add":
                x.add_(skip)
            elif self.skip_method in ("cat", "concat"):
                x = torch.cat([x, skip], dim=1)
            else:
                raise ValueError
            x = decoder_block(x)
            # print(f"Decoder {i}", x.shape)

        # x = self.conv_last(x)
        # print("Last conv", x.shape)
        return x
