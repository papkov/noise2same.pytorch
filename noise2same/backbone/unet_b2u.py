from functools import partial
from typing import Any, List, Tuple, Union

import einops
import torch
from torch import Tensor as T
from torch import nn

Ints = Union[int, Tuple[int, ...]]


class UNetB2U(nn.Module):
    def __init__(
            self,
            in_channels: int,
            base_channels: int = 48,
            kernel_size: int = 3,
            n_dim: int = 2,
            depth: int = 5,
            factor: Union[int, List[Ints]] = 2,
            leak_slope: float = 0.1,
            **kwargs: Any,
    ):
        """
        UNet implementation for Blind2Unblind based on
        https://github.com/zejinwang/Blind2Unblind/blob/main/arch_unet.py

        :param in_channels:
        :param depth:
        :param base_channels:
        :param leak_slope:
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.n_dim = n_dim

        if isinstance(factor, int):
            factor = [factor] * depth
        assert len(factor) == depth, f"factor {factor} must be a list of length depth {depth}"
        self.factor = factor

        conv_block = partial(ConvLeakyRelu, n_dim=n_dim, leak_slope=leak_slope)
        upsample_block = partial(UpsampleBlock, kernel_size=kernel_size, n_dim=n_dim, leak_slope=leak_slope)
        conv_1x1 = partial(nn.Conv2d if n_dim == 2 else nn.Conv3d, kernel_size=1, padding=0, bias=True)
        pool = nn.MaxPool2d if n_dim == 2 else nn.MaxPool3d

        self.head = nn.Sequential(
            conv_block(in_channels, base_channels, kernel_size),
            conv_block(base_channels, base_channels, kernel_size),
        )

        # Encoder
        self.down_path = nn.ModuleList([conv_block(base_channels, base_channels, kernel_size) for _ in range(depth)])
        # If factors are configured in ListConfig, convert them to tuples
        self.pool_path = nn.ModuleList([pool(f if isinstance(f, int) else tuple(f)) for f in factor])

        # Decoder
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            concat_channels = base_channels
            if i == depth - 1:
                concat_channels = 0
            elif i == 0:
                concat_channels = in_channels
            self.up_path.append(upsample_block(base_channels * 2 + concat_channels,
                                               base_channels * 2,
                                               factor=factor[i]))

        # TODO consider moving to a separate module
        self.last = nn.Sequential(
            conv_block(2 * base_channels, 2 * base_channels, 1),
            conv_block(2 * base_channels, 2 * base_channels, 1),
            conv_1x1(2 * base_channels, in_channels),
        )

    def forward(self, x):
        blocks = [x]
        x = self.head(x)
        for i, (down, pool) in enumerate(zip(self.down_path, self.pool_path)):
            x = pool(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        out = self.last(x)
        return out


class ConvLeakyRelu(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            n_dim: int = 2,
            leak_slope: float = 0.1,
    ):
        super().__init__()
        conv = partial(nn.Conv2d if n_dim == 2 else nn.Conv3d,
                       kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.block = nn.Sequential(
            conv(in_channels, out_channels),
            nn.LeakyReLU(leak_slope, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            n_dim: int = 2,
            leak_slope: float = 0.1,
            factor: Union[int, Tuple[int, ...]] = 2,
    ):
        super(UpsampleBlock, self).__init__()
        if isinstance(factor, int):
            factor = (factor,) * n_dim
        assert len(factor) == n_dim, f"factor {factor} must be a tuple of length n_dim {n_dim}"
        self.factor = factor
        conv_block = partial(ConvLeakyRelu, kernel_size=kernel_size, n_dim=n_dim, leak_slope=leak_slope)
        self.block = nn.Sequential(
            conv_block(in_channels, out_channels),
            conv_block(out_channels, out_channels),
        )

    def upsample(self, x: T) -> T:
        factors = {f'u{i}': f for i, f in enumerate(self.factor)}
        in_dims = ['d', 'h', 'w'][-len(self.factor):]
        assert len(in_dims) == len(self.factor)
        out_dims = [f"({u} {d})" for u, d in zip(factors, in_dims)]
        x = einops.repeat(x, f"b c {' '.join(in_dims)} -> b c {' '.join(out_dims)}", **factors)
        return x

    def forward(self, x: T, skip: T) -> T:
        x = self.upsample(x)
        x = torch.cat([x, skip], 1)
        x = self.block(x)
        return x
