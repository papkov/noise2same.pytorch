from enum import Enum
from typing import Tuple, Optional, Iterable
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch
import einops
from timm.models.layers import to_2tuple, trunc_normal_


def concat_shortcut(layer: nn.Module, x: Tensor, y: Tensor):
    if len(y.shape) != len(x.shape):
        return x + y
    x = torch.cat([x, y], -1)
    return layer(x)


class SwinIAMode(Enum):

    DILATED = 0,
    SHUFFLED = 1


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

    def forward(self, x):
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
        mode: SwinIAMode = SwinIAMode.DILATED,
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
        self.mode = mode
        if mode == SwinIAMode.DILATED:
            self.k = MLP(embed_dim, embed_dim)
            self.v = MLP(embed_dim, embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.shortcut = MLP(embed_dim // num_heads * 2, embed_dim // num_heads)

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

    def head_partition(self, x):
        if x.shape[-1] != self.embed_dim:
            return x
        return einops.rearrange(x, 'nw n (nh ch) -> nw nh n ch', nh=self.num_heads)

    def head_partition_reversed(self, x):
        return einops.rearrange(x, "nw nh (ws1 ws2) ch -> nw ws1 ws2 (nh ch)", ws1=self.window_size[0])

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor
    ):
        if self.mode == SwinIAMode.DILATED:
            key = self.k(key)
            value = self.v(value)
        query, key, value = map(self.head_partition, (query, key, value))
        query = query * self.scale
        attn = torch.einsum("...ik,...jk->...ij", query, key)
        relative_position_bias = einops.rearrange(
            self.relative_position_bias_table[self.relative_position_index],
            "(np1 np2) nh -> 1 nh np1 np2", np1=self.num_patches
        )
        attn = attn + relative_position_bias
        batched = len(mask.shape) == 4
        num_windows = mask.shape[1] if batched else mask.shape[0]
        attn = einops.rearrange(attn, "(b nw) ... -> b nw ...", nw=num_windows)
        batch_dimension = 'b nw' if batched else '(b nw)'
        attn += einops.rearrange(mask, f"{batch_dimension} np1 np2 -> b nw 1 np1 np2", nw=num_windows).to(attn.device)
        attn = einops.rearrange(attn, "b nw ... -> (b nw) ...")

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        query = concat_shortcut(self.shortcut, attn @ value, query)
        query, key, value = map(self.head_partition_reversed, (query, key, value))
        query = self.norm(query)
        if self.mode == SwinIAMode.DILATED:
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
        stride: int = 1,
        input_size: Tuple[int] = (128, 128),
        attn_drop: float = 0.05,
        proj_drop: float = 0.05,
        mode: SwinIAMode = SwinIAMode.DILATED
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = MLP(embed_dim, embed_dim, n_layers=2)
        self.mode = mode
        self.attn = DiagWinAttention(
            embed_dim * stride ** 2 if mode == SwinIAMode.SHUFFLED else embed_dim,
            to_2tuple(window_size), mode,
            num_heads, attn_drop, proj_drop
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.window_size = window_size
        self.shift_size = np.array(shift_size) if mode == SwinIAMode.SHUFFLED else np.array(shift_size) * stride
        self.stride = stride
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.attn_mask = self.calculate_mask(input_size)
        self.shortcut = MLP(embed_dim * 2, embed_dim)

    def shift_image(self, x: Optional[Tensor]):
        if x.shape[-1] != self.embed_dim or np.all(self.shift_size == 0):
            return x
        else:
            return torch.roll(x, shifts=tuple(-self.shift_size), dims=(1, 2))

    def shift_image_reversed(self, x: Optional[Tensor]):
        if np.all(self.shift_size == 0):
            return x
        return torch.roll(x, shifts=tuple(self.shift_size), dims=(1, 2))

    def window_partition(self, x: Optional[Tensor]):
        if len(x.shape) == 3:
            return x
        return einops.rearrange(x, 'b (h wh) (w ww) c -> (b h w) (wh ww) c', wh=self.window_size, ww=self.window_size)

    def window_partition_reversed(self, x: Optional[Tensor], x_size: Iterable[int]):
        height, width = x_size
        h, w = height // self.window_size, width // self.window_size
        return einops.rearrange(x, '(b h w) wh ww c -> b (h wh) (w ww) c', h=h, w=w)

    def strided_window_partition(self, x: Optional[Tensor]):
        if len(x.shape) == 3:
            return x
        expression = 'b (h wh sh) (w ww sw) c -> (b h w) (wh ww) (c sh sw)' if self.mode == SwinIAMode.SHUFFLED else \
                     'b (h wh sh) (w ww sw) c -> (b h w sh sw) (wh ww) c'
        return einops.rearrange(x, expression, wh=self.window_size, ww=self.window_size, sh=self.stride, sw=self.stride)

    def strided_window_partition_reversed(self, x: Optional[Tensor], x_size: Iterable[int]):
        height, width = x_size
        h, w = height // self.window_size // self.stride, width // self.window_size // self.stride
        expression = '(b h w) wh ww (c sh sw) -> b (h wh sh) (w ww sw) c' if self.mode == SwinIAMode.SHUFFLED else \
                     '(b h w sh sw) wh ww c -> b (h wh sh) (w ww sw) c'
        return einops.rearrange(x, expression, h=h, w=w, sh=self.stride, sw=self.stride)

    def calculate_mask(self, x_size):
        if self.mode == SwinIAMode.SHUFFLED:
            x_size = [s // self.stride for s in x_size]
        attn_mask = torch.zeros((1, *x_size, 1))
        if np.any(self.shift_size != 0):
            h_slices = (slice(0, -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[:, h, w, :] = cnt
                    cnt += 1
        attn_mask = self.window_partition(attn_mask) if self.mode == SwinIAMode.SHUFFLED else \
                    self.strided_window_partition(attn_mask)
        attn_mask = einops.rearrange(attn_mask, "nw np 1 -> nw 1 np") - attn_mask
        torch.diagonal(attn_mask, dim1=-2, dim2=-1).fill_(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-10 ** 9))
        return attn_mask

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ):
        image_size = key.shape[1:-1]
        if query.shape[-1] == self.embed_dim:
            query = self.norm1(query)
        query, key, value = map(self.shift_image, (query, key, value))
        query, key, value = map(self.strided_window_partition, (query, key, value))
        mask = self.attn_mask if image_size == self.input_size else self.calculate_mask(image_size).to(key.device)
        query, key, value = self.attn(query, key, value, mask=mask)
        query, key, value = map(self.strided_window_partition_reversed, (query, key, value), [image_size] * 3)
        query, key, value = map(self.shift_image_reversed, (query, key, value))
        query = concat_shortcut(self.shortcut, query, self.mlp(self.norm2(query)))
        return query, key, value


class ResidualGroup(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        window_size: int = 8,
        depth: int = 6,
        num_heads: int = 6,
        stride: int = 1,
        mode: SwinIAMode = SwinIAMode.DILATED
    ):
        super().__init__()
        shift_size = window_size // 2
        shifts = ((0, 0), (0, shift_size), (shift_size, shift_size), (shift_size, 0))
        self.blocks = nn.ModuleList([
            BlindSpotBlock(
                embed_dim=embed_dim,
                window_size=window_size,
                shift_size=shifts[i % 4],
                num_heads=num_heads,
                stride=stride,
                mode=mode
            ) for i in range(depth)
        ])
        self.mlp = MLP(embed_dim, embed_dim)
        self.shortcut = MLP(embed_dim * 2, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor
    ):
        shortcut = query
        for block in self.blocks:
            query, key, value = block(query, key, value)
        query = self.mlp(query)
        if query.shape[-1] == shortcut.shape[-1]:
            query = concat_shortcut(self.shortcut, query, shortcut)
        return query, key, value


class SwinIA(nn.Module):

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 96,
        window_size: int = 8,
        depths: Tuple[int] = (6, 6),
        num_heads: Tuple[int] = (6, 6),
        strides: Tuple[int] = (1, 1),
        mode: str = 'dilated'
    ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.mode = SwinIAMode.DILATED if mode == 'dilated' else SwinIAMode.SHUFFLED
        self.embed_k = MLP(in_chans, embed_dim)
        self.embed_v = MLP(in_chans, embed_dim)
        self.proj_last = nn.Linear(embed_dim, in_chans)
        self.shortcut = MLP(embed_dim * 2, embed_dim)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, window_size ** 2, embed_dim // num_heads[0]))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.groups = nn.ModuleList([
            ResidualGroup(
                embed_dim=embed_dim,
                window_size=window_size,
                depth=d,
                num_heads=n,
                stride=s,
                mode=self.mode,
            ) for i, (d, n, s) in enumerate(zip(depths, num_heads, strides))
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = einops.rearrange(x, 'b c ... -> b ... c')
        k = self.embed_k(x)
        v = self.embed_v(x)
        wh, ww = x.shape[1] // self.window_size, x.shape[2] // self.window_size
        full_pos_embed = einops.repeat(self.absolute_pos_embed, "1 (ws1 ws2) ch -> 1 (wh ws1) (ww ws2) (nh ch)",
                                       ws1=self.window_size, nh=self.num_heads[0], wh=wh, ww=ww)
        q = self.absolute_pos_embed
        k = k + full_pos_embed
        v = v + full_pos_embed
        shortcuts = []
        mid = len(self.groups) // 2
        for i, group in enumerate(self.groups):
            q, k, v = group(q, k, v)
            if i < mid:
                shortcuts.append((q, k, v))
            elif shortcuts:
                (q_, k_, v_) = shortcuts.pop()
                q, k, v = map(concat_shortcut, [self.shortcut] * 3, (q, k, v), (q_, k_, v_))
        q = self.proj_last(q)
        q = einops.rearrange(q, 'b ... c -> b c ...')
        return q
