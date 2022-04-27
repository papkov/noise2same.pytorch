# translated from
# https://github.com/divelab/Noise2Same/blob/main/network.py
# https://github.com/divelab/Noise2Same/blob/main/resnet_module.py
from functools import partial
from typing import Any, Tuple

import torch
from torch import Tensor as T
from torch import nn
from torch.nn.functional import normalize

from noise2same.ffc import BN_ACT_FFC, FFC, divide_channels


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
        enable_lfu: bool = True,
        global_ratio: float = 0.5,
        **kwargs: Any,
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
                BN_ACT_FFC,
                n_dim=n_dim,
                ratio_gin=global_ratio,
                ratio_gout=0,
                bn_act_first=True,
                enable_lfu=enable_lfu,
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
                enable_lfu=enable_lfu,
                kernel_size=3,
                padding=1,
                n_dim=n_dim,
                bn_act_first=True,
            )
            self.layers = nn.Sequential(
                bnactffc(
                    ratio_gin=global_ratio,
                    ratio_gout=global_ratio,
                    in_channels=in_channels,
                    out_channels=out_channels,
                ),
                bnactffc(
                    ratio_gin=global_ratio,
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


class ResidualUnitExtraLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_dim: int = 2,
        kernel_size: int = 3,
        downsample: bool = False,
        ffc: bool = False,
        global_ratio: float = 0.5,
        enable_lfu: bool = True,
    ):
        super().__init__()
        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        stride = 2 if downsample else 1
        conv_shortcut = conv
        self.act = nn.ReLU(inplace=True)
        self.conv_shortcut = conv_shortcut(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )
        if ffc:

            bnactffc = partial(
                BN_ACT_FFC,
                stride=1,
                activation_layer=nn.ReLU,
                enable_lfu=enable_lfu,
                kernel_size=3,
                padding=1,
                n_dim=n_dim,
                bn_act_first=True,
            )
            self.layers = nn.Sequential(
                bnactffc(
                    ratio_gin=0,
                    ratio_gout=global_ratio,
                    in_channels=in_channels,
                    out_channels=out_channels,
                ),
                bnactffc(
                    ratio_gin=global_ratio,
                    ratio_gout=global_ratio,
                    in_channels=out_channels,
                    out_channels=out_channels,
                ),
                bnactffc(
                    ratio_gin=global_ratio,
                    ratio_gout=0,
                    in_channels=out_channels,
                    out_channels=out_channels,
                ),
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


class ResidualUnitDualPass(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_dim: int = 2,
        kernel_size: int = 3,
        downsample: bool = False,
        ffc: bool = False,
        last_block: bool = False,
        first_block: bool = False,
        global_ratio: float = 0.5,
        enable_lfu: bool = True,
    ):
        super().__init__()
        self.last_block = last_block
        self.ffc = ffc

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        stride = 2 if downsample else 1
        conv_shortcut = conv

        (
            in_channels_local,
            out_channels_local,
            in_channels_global,
            out_channels_global,
        ) = divide_channels(in_channels, out_channels, global_ratio)
        self.conv_shortcut_local = conv_shortcut(
            in_channels=in_channels if first_block else in_channels_local,
            out_channels=out_channels if last_block else out_channels_local,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )
        if first_block or last_block:
            self.conv_shortcut_global = nn.Identity()
        else:
            self.conv_shortcut_global = conv_shortcut(
                in_channels=in_channels_global,
                out_channels=out_channels_global,
                kernel_size=1,
                padding=0,
                stride=stride,
                bias=False,
            )
        if self.ffc:

            bnactffc = partial(
                BN_ACT_FFC,
                ratio_gin=global_ratio,
                ratio_gout=global_ratio,
                stride=1,
                activation_layer=nn.ReLU,
                enable_lfu=enable_lfu,
                kernel_size=3,
                padding=1,
                n_dim=n_dim,
                bn_act_first=True,
            )
            self.layers = nn.Sequential(
                bnactffc(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    ratio_gin=0 if first_block else global_ratio,
                ),
                bnactffc(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    ratio_gout=0 if last_block else global_ratio,
                ),
            )
        else:
            raise ValueError("Dual pass possible only with ffc=True")

    def forward(self, x: T) -> T:

        if self.ffc:
            # device = x[0].device if isinstance(x, tuple) else x.device
            x_l, x_g = (
                x if type(x) is tuple else (x, 0)
            )  # torch.tensor(0, device=device))
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
        unit_type: str = "default",
        last_block: bool = False,
        first_block: bool = False,
        global_ratio: float = 0.5,
        enable_lfu: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.block_size = block_size

        if unit_type == "default":
            unit = ResidualUnit
        elif unit_type == "extra_layer":
            unit = ResidualUnitExtraLayer
        elif unit_type == "dual_pass":
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
                    last_block=last_block,
                    first_block=first_block,
                    global_ratio=global_ratio,
                    enable_lfu=enable_lfu,
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
        unit_type: str = "default",
        global_ratio: float = 0.5,
        enable_lfu: bool = True,
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
            conv = partial(
                FFC,
                n_dim=n_dim,
                ratio_gin=0,
                ratio_gout=global_ratio,
                enable_lfu=enable_lfu,
            )

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
            if unit_type == "dual_pass" and ffc == True:
                self.downsampling_block = TwinSample(
                    downsample, in_channels, out_channels, global_ratio
                )
            else:
                self.downsampling_block = downsample(
                    in_channels=in_channels, out_channels=out_channels
                )
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
            unit_type=unit_type,
            global_ratio=global_ratio,
            enable_lfu=enable_lfu,
        )

    def forward(self, x: T) -> T:

        x = self.downsampling_block(x)
        x = self.resblock(x)
        return x


class TwinSample(nn.Module):
    """For up and downsampling when there is both local and global branch present"""

    def __init__(self, unit, in_channels, out_channels, global_ratio):
        super().__init__()
        (
            in_channels_local,
            out_channels_local,
            in_channels_global,
            out_channels_global,
        ) = divide_channels(in_channels, out_channels, global_ratio)

        self.sample_local = unit(
            in_channels=in_channels_local, out_channels=out_channels_local
        )  # local
        self.sample_global = unit(
            in_channels=in_channels_global, out_channels=out_channels_global
        )  # global

    def forward(self, x: Tuple[T,T]):
        assert isinstance(x, tuple)
        x_l, x_g = x
        x_l = self.sample_local(x_l)
        x_g = self.sample_global(x_g)
        return (x_l, x_g)


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
        unit_type: str = "default",
        global_ratio: float = 0.5,
        enable_lfu: bool = True,
        ffc_enc: bool = True,
        ffc_dec: bool = True,
        ffc_bottom: bool = True,
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
        self.ffc_enc = ffc_enc
        self.ffc_dec = ffc_dec
        self.ffc_bottom = ffc_bottom
        self.unit_type = unit_type
        self.global_ratio = global_ratio
        print(f"Use {self.skip_method} skip method")

        if ffc_enc and unit_type != "extra_layer":  # only used for first conv
            conv = partial(
                FFC,
                n_dim=n_dim,
                ratio_gin=0,
                ratio_gout=global_ratio,
                enable_lfu=enable_lfu,
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
                    ffc=ffc_enc,
                    unit_type="default"
                    if ffc_enc == False and unit_type == "dual_pass"
                    else unit_type,
                    global_ratio=global_ratio,
                    enable_lfu=enable_lfu,
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
                    ffc=ffc_enc,
                    unit_type="default"
                    if ffc_enc == False and unit_type == "dual_pass"
                    else unit_type,
                    global_ratio=global_ratio,
                    enable_lfu=enable_lfu,
                )
            )

        # Bottom block
        self.bottom_block = ResidualBlock(
            in_channels=out_channels,
            out_channels=base_channels * (2 ** (depth - 1)),
            n_dim=n_dim,
            kernel_size=kernel_size,
            block_size=1,
            ffc=True if ffc_enc or ffc_dec is True else False,
            unit_type=unit_type,
            global_ratio=global_ratio,
            enable_lfu=enable_lfu,
            last_block=True if (ffc_enc == True and ffc_dec == False) else False,
            first_block=True if (ffc_enc == False and ffc_dec == True) else False,
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        for i in range(self.depth - 1, 0, -1):

            in_channels = int(base_channels * (2 ** i))
            out_channels = int(base_channels * (2 ** (i - 1)))

            if upsampling[i - 1] == "conv":
                upsample = partial(conv_transpose, kernel_size=2, stride=2, bias=True)
                if unit_type == "dual_pass" and ffc_dec == True:

                    upsampling_block = TwinSample(
                        upsample, in_channels, out_channels, global_ratio
                    )

                else:
                    upsampling_block = upsample(
                        in_channels=in_channels, out_channels=out_channels
                    )
            else:
                raise ValueError(
                    f"Upsampling method {upsampling[i - 1]} not supported for {n_dim}D"
                )

            self.upsampling_blocks.append(upsampling_block)

            # Here goes skip connection, then decoder block
            self.decoder_blocks.append(
                ResidualBlock(
                    in_channels=out_channels
                    * (2 if self.skip_method != "add" else 1),  # *2
                    out_channels=out_channels,
                    n_dim=n_dim,
                    kernel_size=kernel_size,
                    block_size=decoding_block_sizes[depth - 1 - i],
                    ffc=ffc_dec if unit_type != "default" else False,
                    unit_type="default"
                    if ffc_dec == False and unit_type == "dual_pass"
                    else unit_type,
                    last_block=True if i == 1 and unit_type == "dual_pass" else False,
                    global_ratio=global_ratio,
                    enable_lfu=enable_lfu,
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

        for (i, (upsampling_block, decoder_block, skip),) in enumerate(
            zip(self.upsampling_blocks, self.decoder_blocks, encoder_outputs[::-1],)
        ):
            x = upsampling_block(x)
            if self.unit_type == "dual_pass":  # when the skip is tuple
                (x_l, x_g) = x if type(x) is tuple else (x, 0)
                if self.skip_method == "add":
                    # enc dec paramterization not supported
                    x_l.add_(skip[0])
                    x_g.add_(skip[1])
                elif self.skip_method in ("cat", "concat"):

                    if self.ffc_enc == True and self.ffc_dec == False:
                        combined_skip = torch.cat([skip[0], skip[1]], dim=1)
                        x = torch.cat([x, combined_skip], dim=1)
                    elif (
                        self.ffc_enc == False and self.ffc_dec == True
                    ):  # probably not worth it
                        n, c, *s = skip.shape  # s is (d,h,w) or (h,w)
                        global_channels = int(c * self.global_ratio)
                        local_channels = c - global_channels
                        skip_split = torch.split(
                            skip, [local_channels, global_channels], dim=1
                        )  # channel dimension
                        x_l = torch.cat([x_l, skip_split[0]], dim=1)
                        x_g = torch.cat([x_g, skip_split[1]], dim=1)
                        x = (x_l, x_g)
                    else:
                        x_l = torch.cat([x_l, skip[0]], dim=1)
                        x_g = torch.cat([x_g, skip[1]], dim=1)
                        x = (x_l, x_g)
            else:

                if self.skip_method == "add":
                    x.add_(skip)
                elif self.skip_method in ("cat", "concat"):
                    x = torch.cat([x, skip], dim=1)

            x = decoder_block(x)

        return x
