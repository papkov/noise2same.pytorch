import itertools
from typing import Any, Tuple, Optional, Iterable, Set

import torch
import torch.nn as nn
from torch import Tensor as T
from timm.models.layers import to_2tuple, trunc_normal_
import numpy as np
import einops


def connect_shortcut(layer: nn.Module, x: T, y: T) -> T:
    x = torch.cat([x, y], -1)
    return layer(x)


class MLP(nn.Module):

    def __init__(
            self,
            in_features: int = 96,
            out_features: int = 96,
            n_layers: int = 1,
            hidden_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop=0.,
    ):
        super().__init__()
        hidden_features = hidden_features or out_features
        features = [hidden_features] * (n_layers + 1)
        features[0], features[-1] = in_features, out_features
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(features[i], features[i + 1]),
            ) for i in range(n_layers)
        ])
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: T) -> T:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
            x = self.drop(x)
        return x


class DiagonalWindowAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        window_size: Tuple[int] = (8, 8),
        dilation: int = 1,
        shuffle: int = 1,
        num_heads: int = 6,
        attn_drop: float = 0.05,
        proj_drop: float = 0.05,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.window_size = window_size
        self.num_patches = np.prod(window_size).item()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dilation = dilation
        self.shuffle = shuffle

        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5 / shuffle
        # todo bias=False?
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_q = nn.LayerNorm([shuffle ** 2, embed_dim])
        self.norm_k = nn.LayerNorm([shuffle ** 2, embed_dim])
        self.norm_v = nn.LayerNorm([shuffle ** 2, embed_dim])

        window_bias_shape = [2 * s - 1 for s in window_size]
        self.relative_position_bias_table = nn.Parameter(torch.zeros(np.prod(window_bias_shape).item(), num_heads))

        coords = torch.stack(torch.meshgrid([torch.arange(s) for s in window_size], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = torch.einsum("n...->...n", relative_coords)

        coefficients = [np.prod(window_bias_shape[i:]) for i in range(1, len(window_size))]
        for dim, size in enumerate(window_size):
            relative_coords[..., dim] += size - 1
            if dim < len(window_size) - 1:
                relative_coords[..., dim] *= coefficients[dim]

        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def group_partition(self, x: T) -> T:
        return einops.rearrange(x, 'nw (wh sh ww sw) ch -> nw (wh ww) (sh sw) ch',
                                wh=self.window_size[0], sh=self.shuffle, sw=self.shuffle)

    def group_partition_reversed(self, x: T) -> T:
        return einops.rearrange(x, "nw (wh ww) (sh sw) ch -> nw (wh sh ww sw) ch",
                                wh=self.window_size[0], sh=self.shuffle)

    def head_partition(self, x: T) -> T:
        return einops.rearrange(x, 'nw ... (nh ch) -> nw nh ... ch', nh=self.num_heads)

    def head_partition_reversed(self, x: T) -> T:
        return einops.rearrange(x, "nw nh ... ch -> nw ... (nh ch)")

    def forward(
        self,
        query: T,
        key: T,
        value: T,
        mask: T,
    ) -> T:
        query, key, value = map(self.group_partition, (query, key, value))
        query, key, value = self.norm_q(query), self.norm_k(key), self.norm_v(value)
        query, key, value = map(self.head_partition, (query, key, value))
        query = query * self.scale
        attn = torch.einsum("...qsc,...ksc->...qk", query, key)
        relative_position_bias = einops.rearrange(
            self.relative_position_bias_table[self.relative_position_index],
            "(np1 np2) nh -> 1 nh np1 np2", np1=self.num_patches
        )
        attn = attn + relative_position_bias
        attn += einops.repeat(mask, f"nw np1 np2 -> (b nw) 1 np1 np2", b=attn.shape[0] // mask.shape[0]).to(attn.device)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        query = torch.einsum("...qk,...ksc->...qsc", attn, value)
        query, key, value = map(self.head_partition_reversed, (query, key, value))
        query, key, value = map(self.group_partition_reversed, (query, key, value))
        query = self.proj(query)
        query = self.proj_drop(query)
        return query


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        window_size: int = 8,
        shift_size: Tuple[int] = (0, 0),
        num_heads: int = 6,
        dilation: int = 1,
        shuffle: int = 1,
        input_size: Tuple[int] = (128, 128),
        attn_drop: float = 0.05,
        proj_drop: float = 0.05,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = MLP(embed_dim, embed_dim, n_layers=2)
        self.attn = DiagonalWindowAttention(
            embed_dim, to_2tuple(window_size), dilation, shuffle, num_heads,
            attn_drop, proj_drop
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.window_size = window_size
        self.shift_size = np.array(shift_size)
        self.dilation = dilation
        self.shuffle = shuffle
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.attn_mask = self.calculate_mask(input_size)

    def shift_image(self, x: T) -> T:
        if np.all(self.shift_size == 0):
            return x
        return torch.roll(x, shifts=tuple(-self.shift_size * self.dilation * self.shuffle), dims=(1, 2))

    def shift_image_reversed(self, x: T) -> T:
        if np.all(self.shift_size == 0):
            return x
        return torch.roll(x, shifts=tuple(self.shift_size * self.dilation * self.shuffle), dims=(1, 2))

    def window_partition(self, x: T) -> T:
        return einops.rearrange(x, 'b (h wh sh dh) (w ww sw dw) c -> (b h w dh dw) (wh sh ww sw) c',
                                wh=self.window_size, ww=self.window_size, sh=self.shuffle, sw=self.shuffle,
                                dh=self.dilation, dw=self.dilation)

    def window_partition_reversed(self, x: T, x_size: Iterable[int]) -> T:
        height, width = x_size
        h = height // (self.window_size * self.shuffle * self.dilation)
        w = width // (self.window_size * self.shuffle * self.dilation)
        return einops.rearrange(x, '(b h w dh dw) (wh sh ww sw) c -> b (h wh sh dh) (w ww sw dw) c',
                                h=h, w=w, sh=self.shuffle, sw=self.shuffle, wh=self.window_size,
                                dh=self.dilation, dw=self.dilation)

    def mask_window_partition(self, mask: T) -> T:
        return einops.rearrange(mask, 'b (h wh dh) (w ww dw) c -> (b h w dh dw) (wh ww) c',
                                wh=self.window_size, ww=self.window_size, dh=self.dilation, dw=self.dilation)

    def calculate_mask(self, x_size: Iterable[int]) -> T:
        x_size = [s // self.shuffle for s in x_size]
        attn_mask = torch.zeros((1, *x_size, 1))
        if np.any(self.shift_size != 0):
            slices = [(slice(0, s), slice(s, None)) for s in -self.shift_size * self.dilation]
            cnt = 0
            for h, w in itertools.product(*slices):
                attn_mask[:, h, w, :] = cnt
                cnt += 1
        attn_mask = self.mask_window_partition(attn_mask)
        attn_mask = einops.rearrange(attn_mask, "nw np 1 -> nw 1 np") - attn_mask
        torch.diagonal(attn_mask, dim1=-2, dim2=-1).fill_(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -10 ** 9)
        return attn_mask

    def forward(
        self,
        query: T,
        key: T,
        value: T,
    ) -> T:
        image_size = key.shape[1:-1]
        shortcut = query
        query, key, value = map(self.shift_image, (query, key, value))
        query, key, value = map(self.window_partition, (query, key, value))
        mask = self.attn_mask if image_size == self.input_size else self.calculate_mask(image_size).to(key.device)
        query = self.attn(query, key, value, mask=mask)
        query, key, value = map(self.window_partition_reversed, (query, key, value), [image_size] * 3)
        query, key, value = map(self.shift_image_reversed, (query, key, value))
        query = query + shortcut
        query = query + self.mlp(self.norm(query))
        return query


class ResidualGroup(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        window_size: int = 8,
        depth: int = 6,
        num_heads: int = 6,
        dilation: int = 1,
        shuffle: int = 1,
        input_size: Tuple[int] = (128, 128),
    ):
        super().__init__()
        shift_size = window_size // 2
        shifts = ((0, 0), (0, shift_size), (shift_size, shift_size), (shift_size, 0))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                window_size=window_size,
                shift_size=shifts[i % 4],
                num_heads=num_heads,
                dilation=dilation,
                shuffle=shuffle,
                input_size=input_size,
            ) for i in range(depth)
        ])

    def forward(
        self,
        query: T,
        key: T,
        value: T,
    ) -> T:
        for block in self.blocks:
            query = block(query, key, value)
        return query


class SwinIA(nn.Module):

    def __init__(
            self,
            in_channels: int = 1,
            embed_dim: int = 96,
            window_size: int = 8,
            input_size: int = 128,
            depths: Tuple[int] = (8, 4, 4, 4, 4, 4, 8),
            num_heads: Tuple[int] = (6, 6, 6, 6, 6, 6, 6),
            dilations: Tuple[int] = (1, 1, 1, 1, 1, 1, 1),
            shuffles: Tuple[int] = (1, 1, 1, 1, 1, 1, 1),
            **kwargs: Any,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_k = MLP(in_channels, embed_dim)
        self.embed_v = MLP(in_channels, embed_dim)
        self.proj_last = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, in_channels)
        )
        self.shortcut1 = nn.Linear(embed_dim * 2, embed_dim)
        self.shortcut2 = nn.Linear(embed_dim * 2, embed_dim)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(window_size ** 2, embed_dim // num_heads[0]))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.groups = nn.ModuleList([
            ResidualGroup(
                embed_dim=embed_dim,
                window_size=window_size,
                depth=d,
                num_heads=n,
                dilation=dl,
                shuffle=sh,
                input_size=to_2tuple(input_size)
            ) for i, (d, n, dl, sh) in enumerate(zip(depths, num_heads, dilations, shuffles))
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) and m.bias is not None and m.weight is not None:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> Set[str]:
        return {'relative_position_bias_table'}

    def forward(self, x: T) -> T:
        x = einops.rearrange(x, 'b c ... -> b ... c')
        k = self.embed_k(x)
        v = self.embed_v(x)
        wh, ww = x.shape[1] // self.window_size, x.shape[2] // self.window_size
        full_pos_embed = einops.repeat(self.absolute_pos_embed, "(ws1 ws2) ch -> b (wh ws1) (ww ws2) (nh ch)",
                                       b=x.shape[0], ws1=self.window_size, wh=wh, ww=ww, nh=self.num_heads[0])
        q, k, v = full_pos_embed, k + full_pos_embed, v + full_pos_embed
        shortcuts = []
        mid = len(self.groups) // 2
        for i, group in enumerate(self.groups):
            if i < mid:
                q_ = group(q, k, v)
                shortcuts.append(q_)
            else:
                q = group(q, k, v)
                if shortcuts:
                    q = connect_shortcut(self.shortcut1 if len(shortcuts) == 1 else self.shortcut2, q, shortcuts.pop())
        q = self.proj_last(q)
        q = einops.rearrange(q, 'b ... c -> b c ...')
        return q
