import itertools
from typing import Tuple, Optional, Iterable, Set
from torch import Tensor as T
import torch.nn as nn
import numpy as np
import torch
import einops
from timm.models.layers import to_2tuple, trunc_normal_


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
                nn.LayerNorm(features[i + 1])
            ) for i in range(n_layers)
        ])
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: T) -> T:
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
            x = self.drop(x)
        return x


class DiagWinAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        window_size: Tuple[int] = (8, 8),  # todo: something about tuple window size
        dilation: int = 1,
        shuffle: int = 1,
        num_heads: int = 6,
        attn_drop: float = 0.05,
        proj_drop: float = 0.05,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.LayerNorm(embed_dim)
        self.window_size = window_size
        self.num_patches = np.prod(window_size).item()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scale = (embed_dim // num_heads) ** -0.5
        self.dilation = dilation
        self.shuffle = shuffle
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.k = MLP(embed_dim, embed_dim)
        self.v = MLP(embed_dim, embed_dim)
        self.shortcut = MLP(embed_dim * 2, embed_dim)

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

    def head_partition(self, x: T) -> T:
        return einops.rearrange(x, 'nw (wh sh ww sw) (nh ch) -> nw nh (wh ww) (sh sw) ch',
                                wh=self.window_size[0], sh=self.shuffle, sw=self.shuffle, nh=self.num_heads)

    def head_partition_reversed(self, x: T) -> T:
        return einops.rearrange(x, "nw nh (wh ww) (sh sw) ch -> nw (wh sh ww sw) (nh ch)",
                                wh=self.window_size[0], sh=self.shuffle)

    def forward(
        self,
        query: T,
        key: T,
        value: T,
        mask: T,
        is_masked: bool = True
    ) -> Tuple[T, T, T]:
        shortcut = query
        query, key, value = map(self.head_partition, (query, key, value))
        query = query * self.scale
        attn = torch.einsum("...qsc,...ksc->...qk", query, key)
        relative_position_bias = einops.rearrange(
            self.relative_position_bias_table[self.relative_position_index],
            "(np1 np2) nh -> 1 nh np1 np2", np1=self.num_patches
        )
        attn = attn + relative_position_bias
        if is_masked:
            torch.diagonal(mask, dim1=-2, dim2=-1).fill_(1)
        mask = mask.masked_fill(mask != 0, -10 ** 9)
        attn += einops.repeat(mask, f"nw np1 np2 -> (b nw) 1 np1 np2", b=attn.shape[0] // mask.shape[0]).to(attn.device)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        query = torch.einsum("...qk,...ksc->...qsc", attn, value)
        query, key, value = map(self.head_partition_reversed, (query, key, value))
        query = query + shortcut
        query = self.norm(query)
        query = self.proj(query)
        query = self.proj_drop(query)
        return query, key, value


class BlindSpotBlock(nn.Module):

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
        is_first: bool = False
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = MLP(embed_dim, embed_dim, n_layers=2)
        self.attn = DiagWinAttention(
            embed_dim, to_2tuple(window_size), dilation, shuffle, num_heads,
            attn_drop, proj_drop
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.window_size = window_size
        self.shift_size = np.array(shift_size)
        self.dilation = dilation
        self.shuffle = shuffle
        self.is_first = is_first
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.attn_mask = self.calculate_mask(input_size)
        self.shortcut = MLP(embed_dim * 2, embed_dim)

    def shift_image(self, x: T) -> T:
        if np.all(self.shift_size == 0):
            return x
        return torch.roll(x, shifts=tuple(-self.shift_size * self.dilation * self.shuffle), dims=(1, 2))

    def shift_image_reversed(self, x: T) -> T:
        if np.all(self.shift_size == 0):
            return x
        return torch.roll(x, shifts=tuple(self.shift_size * self.dilation * self.shuffle), dims=(1, 2))

    def strided_window_partition(self, x: T) -> T:
        return einops.rearrange(x, 'b (h wh sh dh) (w ww sw dw) c -> (b h w dh dw) (wh sh ww sw) c',
                                wh=self.window_size, ww=self.window_size, sh=self.shuffle, sw=self.shuffle,
                                dh=self.dilation, dw=self.dilation)

    def strided_window_partition_reversed(self, x: T, x_size: Iterable[int]) -> T:
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
        return attn_mask

    def forward(
        self,
        query: T,
        key: T,
        value: T,
        is_masked: bool = False
    ) -> Tuple[T, T, T]:
        image_size = key.shape[1:-1]
        if not self.is_first:
            query = self.norm1(query)
        shortcut = query
        query, key, value = map(self.shift_image, (query, key, value))
        query, key, value = map(self.strided_window_partition, (query, key, value))
        mask = self.attn_mask if image_size == self.input_size else self.calculate_mask(image_size).to(key.device)
        query, key, value = self.attn(query, key, value, mask=mask, is_masked=is_masked)
        query, key, value = map(self.strided_window_partition_reversed, (query, key, value), [image_size] * 3)
        query, key, value = map(self.shift_image_reversed, (query, key, value))
        query = query + self.mlp(self.norm2(query))
        # query = connect_shortcut(self.shortcut, query, shortcut)
        return query, key, value


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
        is_first: bool = False
    ):
        super().__init__()
        shift_size = window_size // 2
        shifts = ((0, 0), (0, shift_size), (shift_size, shift_size), (shift_size, 0))
        self.is_first = is_first
        self.blocks = nn.ModuleList([
            BlindSpotBlock(
                embed_dim=embed_dim,
                window_size=window_size,
                shift_size=shifts[i % 4],
                num_heads=num_heads,
                dilation=dilation,
                shuffle=shuffle,
                input_size=input_size,
                is_first=is_first and i == 0
            ) for i in range(depth)
        ])
        self.shortcut = MLP(embed_dim * 2, embed_dim)

    def forward(
        self,
        query: T,
        key: T,
        value: T,
        is_masked: bool = True
    ) -> Tuple[T, T, T]:
        shortcut = query
        for block in self.blocks:
            query, _, _ = block(query, key, value, is_masked)
        query = connect_shortcut(self.shortcut, query, shortcut)
        return query, key, value


class SwinIA(nn.Module):

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 96,
        window_size: int = 8,
        input_size: int = 128,
        depths: Tuple[int] = (8, 4, 4, 4, 4, 4, 8),
        num_heads: Tuple[int] = (6, 6, 6, 6, 6, 6, 6),
        dilations: Tuple[int] = (1, 1, 1, 1, 1, 1, 1),
        shuffles: Tuple[int] = (1, 1, 1, 1, 1, 1, 1)
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.embed_k = MLP(in_chans, embed_dim)
        self.embed_v = MLP(in_chans, embed_dim)
        self.proj_last = nn.Linear(embed_dim, in_chans)
        self.shortcut = MLP(embed_dim * 2, embed_dim)
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
                input_size=to_2tuple(input_size),
                is_first=i == 0
            ) for i, (d, n, dl, sh) in enumerate(zip(depths, num_heads, dilations, shuffles))
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> Set[str]:
        return {'relative_position_bias_table'}

    def forward(self, x: T, is_masked: bool = True) -> T:
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
            q, _, _ = group(q, k, v, is_masked)
            # if i < mid:
            #     shortcuts.append(q)
            # elif shortcuts:
            #     q = connect_shortcut(self.shortcut, q, shortcuts.pop())
        q = self.proj_last(q)
        q = einops.rearrange(q, 'b ... c -> b c ...')
        return q
