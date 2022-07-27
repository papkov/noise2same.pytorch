from functools import partial
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor as T


class FFCSE_block(nn.Module):
    def __init__(self, channels: int, ratio_g: float, n_dim: int = 2):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        conv = partial(conv, kernel_size=1, bias=True)
        pool = nn.AdaptiveAvgPool2d if n_dim == 2 else nn.AdaptiveAvgPool3d

        self.avgpool = pool((1,) * n_dim)
        self.conv1 = conv(channels, channels // r)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else conv(channels // r, in_cl)
        self.conv_a2g = None if in_cg == 0 else conv(channels // r, in_cg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Union[T, Tuple[T, T]]) -> Tuple[T, T]:
        x = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x

        x = id_l if isinstance(id_g, int) else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, groups: int = 1, n_dim: int = 2
    ):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.n_dim = n_dim
        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        self.conv_layer = conv(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = bn(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: T) -> T:
        batch, c, *s = x.size()
        dim = (2, 3, 4)[: len(s)]

        # (batch, c, h, w/2+1) complex number
        ffted = torch.fft.rfftn(x.float(), s=s, dim=dim, norm="ortho")
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = torch.tensor_split(ffted, 2, dim=1)
        ffted = torch.complex(ffted[0].float(), ffted[1].float())
        output = torch.fft.irfftn(ffted, s=s, dim=dim, norm="ortho")

        return output


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        enable_lfu: bool = True,
        n_dim: int = 2,
    ):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        conv = partial(conv, kernel_size=1, groups=groups, bias=False)

        bn = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        pool = nn.AvgPool2d if n_dim == 2 else nn.AvgPool3d
        self.downsample = (
            pool(kernel_size=2, stride=stride) if stride == 2 else nn.Identity()
        )

        self.stride = stride
        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels // 2),
            bn(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, n_dim)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups, n_dim)
        self.conv2 = conv(out_channels // 2, out_channels)

    def forward(self, x: T) -> T:

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, *s = x.shape  # s is (d,h,w) or (h,w)
            h = s[len(s) - 2]
            split_no = 2
            split_s = h // split_no
            split = torch.split(x[:, : c // 4], split_s, dim=-2)
            xs = torch.cat(split, dim=1).contiguous()

            next_split = torch.split(xs, split_s, dim=-1)
            xs = torch.cat(next_split, dim=1).contiguous()
            xs = self.lfu(xs)
            rep = (1,) * len(s)
            xs = xs.repeat(*rep, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class SpectralTransformTransposed(SpectralTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        **kwargs,
    ):
        super(SpectralTransformTransposed, self).__init__(
            in_channels, out_channels, **kwargs
        )
        self.downsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")


class FFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        enable_lfu: bool = True,
        n_dim: int = 2,
        **kwargs: Any,
    ):
        super(FFC, self).__init__()

        conv = nn.Conv2d if n_dim == 2 else nn.Conv3d
        conv = partial(
            conv,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.convl2l = (
            nn.Identity() if in_cl == 0 or out_cl == 0 else conv(in_cl, out_cl)
        )
        self.convl2g = (
            nn.Identity() if in_cl == 0 or out_cg == 0 else conv(in_cl, out_cg)
        )
        self.convg2l = (
            nn.Identity() if in_cg == 0 or out_cl == 0 else conv(in_cg, out_cl)
        )
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg,
            out_cg,
            stride,
            1 if groups == 1 else groups // 2,
            enable_lfu,
            n_dim=n_dim,
        )

    def forward(self, x: Union[T, Tuple[T, T]]) -> Tuple[T, T]:
        device = x[0].device if isinstance(x, tuple) else x.device
        x_l, x_g = x if type(x) is tuple else (x, torch.tensor(0, device=device))
        out_xl = torch.tensor(0, device=device)
        out_xg = torch.tensor(0, device=device)

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFCTransposed(FFC):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 2,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        enable_lfu: bool = True,
        n_dim: int = 2,
        mode="nearest",
        **kwargs: Any,
    ):
        super(FFC, self).__init__()

        conv = nn.ConvTranspose2d if n_dim == 2 else nn.ConvTranspose3d
        conv = partial(
            conv,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        self.convl2l = (
            nn.Upsample(scale_factor=stride, mode=mode)
            if in_cl == 0 or out_cl == 0
            else conv(in_cl, out_cl)
        )
        self.convl2g = (
            nn.Upsample(scale_factor=stride, mode=mode)
            if in_cl == 0 or out_cg == 0
            else conv(in_cl, out_cg)
        )
        self.convg2l = (
            nn.Upsample(scale_factor=stride, mode=mode)
            if in_cg == 0 or out_cl == 0
            else conv(in_cg, out_cl)
        )
        self.convg2g = (
            nn.Upsample(scale_factor=stride, mode=mode)
            if in_cg == 0 or out_cg == 0
            else SpectralTransformTransposed(
                in_cg,
                out_cg,
                stride,
                groups=1 if groups == 1 else groups // 2,
                enable_lfu=enable_lfu,
                n_dim=n_dim,
            )
        )


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        activation_layer: nn.Module = nn.Identity,
        enable_lfu: bool = True,
        n_dim: int = 2,
        bn_act_first: bool = False,
    ):
        super(FFC_BN_ACT, self).__init__()

        norm_layer = nn.BatchNorm2d if n_dim == 2 else nn.BatchNorm3d
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            n_dim,
        )

        # todo it works for now but it should be changed to be more explicit in initialization
        if bn_act_first:
            ratio_gout = ratio_gin
            out_channels = in_channels

        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x: Union[T, Tuple[T, T]]) -> Tuple[T, T]:
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class BN_ACT_FFC(FFC_BN_ACT):
    def forward(self, x: Union[T, Tuple[T, T]]) -> Tuple[T, T]:
        x_l, x_g = x if type(x) is tuple else (x, 0)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        x_l, x_g = self.ffc((x_l, x_g))
        return x_l, x_g


def divide_channels(in_channels: int, out_channels: int, global_ratio: float):
    in_channels_global = int(in_channels * global_ratio)
    in_channels_local = in_channels - in_channels_global
    out_channels_global = int(out_channels * global_ratio)
    out_channels_local = out_channels - out_channels_global
    return (
        in_channels_local,
        out_channels_local,
        in_channels_global,
        out_channels_global,
    )
