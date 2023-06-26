# translated from
# https://github.com/divelab/Noise2Same/blob/main/network.py
# https://github.com/divelab/Noise2Same/blob/main/resnet_module.py
import logging
from typing import Tuple, Optional, List, Union

import torch
from torch import Tensor as T
from torch import nn

Ints = Union[int, Tuple[int, ...]]

log = logging.getLogger(__name__)


class RegressionHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, n_dim: int = 2, kernel_size: int = 1):
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
            downsampling_factor: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsampling_factor = downsampling_factor

        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d

        self.act = nn.ReLU(inplace=True)
        # todo parametrize as in the original repo (bn momentum is inverse)
        self.bn = bn(in_channels, momentum=1 - 0.997, eps=1e-5)
        self.conv_shortcut = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=downsampling_factor or 1,
            bias=False,
        )

        self.layers = nn.Sequential(
            conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=downsampling_factor or kernel_size,
                padding=0 if downsampling_factor is not None else kernel_size // 2,
                stride=downsampling_factor or 1,
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
        shortcut = x
        x = self.bn(x)
        x = self.act(x)
        if self.in_channels != self.out_channels or self.downsampling_factor is not None:
            shortcut = self.conv_shortcut(x)
        x = self.layers(x)
        return x + shortcut


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_size: int = 1,
            n_dim: int = 2,
            kernel_size: int = 3,
            downsampling_factor: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsampling_factor = downsampling_factor
        self.block_size = block_size

        self.block = nn.Sequential(
            *[
                ResidualUnit(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    downsampling_factor=downsampling_factor if i == 0 else None,
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
            downsampling_factor: Ints = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.block_size = block_size

        if isinstance(downsampling_factor, int):
            downsampling_factor = (downsampling_factor,) * n_dim
        assert len(downsampling_factor) == n_dim

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d

        if downsampling == "res":
            downsampling_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                n_dim=n_dim,
                kernel_size=kernel_size,
                block_size=1,
                downsampling_factor=downsampling_factor,
            )
        elif downsampling == "conv":
            downsampling_block = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=downsampling_factor,
                stride=downsampling_factor,
                bias=True,
            )
        # TODO pooling
        else:
            raise ValueError("downsampling should be `res`, `conv`")

        self.block = nn.Sequential(
            downsampling_block,
            ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_dim=n_dim,
                block_size=block_size,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x: T) -> T:
        return self.block(x)


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
            downsampling_factor: Union[int, List[Ints]] = 2,
            skip_method: str = "concat",
            **kwargs,
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
        """
        super().__init__()

        assert depth == len(encoding_block_sizes)
        assert encoding_block_sizes[0] > 0
        assert encoding_block_sizes[-1] == 0
        assert depth == len(decoding_block_sizes) + 1
        assert depth == len(downsampling) + 1
        assert skip_method in ["add", "concat", "cat"]

        if isinstance(downsampling_factor, int):
            downsampling_factor = [downsampling_factor] * len(downsampling)
        assert len(downsampling_factor) == len(downsampling)

        self.in_channels = in_channels
        self.n_dim = n_dim
        self.depth = depth
        self.base_channels = base_channels
        self.encoding_block_sizes = encoding_block_sizes
        self.decoding_block_sizes = decoding_block_sizes
        self.downsampling = downsampling
        self.downsampling_factor = downsampling_factor
        self.skip_method = skip_method
        logging.debug(f"Use {self.skip_method} skip method")

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
                    downsampling_factor=downsampling_factor[i - 2],
                )
            )

        # Bottom block
        self.bottom_block = ResidualBlock(
            in_channels=out_channels,
            out_channels=base_channels * (2 ** (depth - 1)),
            n_dim=n_dim,
            kernel_size=kernel_size,
            block_size=1,
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
                    kernel_size=downsampling_factor[i - 1],
                    stride=downsampling_factor[i - 1],
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
        x = self.encoder_blocks[0](x)

        for i, encoder_block in enumerate(self.encoder_blocks[1:]):
            encoder_outputs.append(x)
            x = encoder_block(x)

        x = self.bottom_block(x)

        for i, (upsampling_block, decoder_block, skip) in enumerate(
            zip(self.upsampling_blocks, self.decoder_blocks, encoder_outputs[::-1])
        ):
            x = upsampling_block(x)
            if self.skip_method == "add":
                x.add_(skip)
            elif self.skip_method in ("cat", "concat"):
                x = torch.cat([x, skip], dim=1)
            else:
                raise ValueError
            x = decoder_block(x)

        return x
