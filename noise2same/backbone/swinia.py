from typing import Tuple, Optional, Iterable
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch
import einops
from timm.models.layers import to_2tuple, trunc_normal_


class Conv1x1(nn.Module):

    def __init__(
        self,
        in_features: int = 96,
        out_features: int = 96,
        channels_last: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.channels_last = channels_last
        self.conv = nn.Conv2d(in_features, out_features, 1)

    def forward(self, x):
        if x.shape[1] != self.in_features:
            x = einops.rearrange(x, 'b ... c -> b c ...')
        x = self.conv(x)
        if self.channels_last:
            return einops.rearrange(x, 'b c ... -> b ... c')
        return x


class MLP(nn.Module):

    def __init__(
        self,
        in_features: int = 96,
        out_features: int = 96,
        two_layers: bool = True,
        hidden_features: Optional[int] = None,
        layer: nn.Module = nn.Linear,
        act_layer: nn.Module = nn.GELU,
        drop=0.,
    ):
        super().__init__()
        hidden_features = hidden_features or out_features
        self.layer1 = layer(in_features, hidden_features)
        self.bn1 = nn.LayerNorm(hidden_features)
        self.two_layers = two_layers
        if two_layers:
            self.layer2 = layer(hidden_features, out_features)
            self.bn2 = nn.LayerNorm(out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.two_layers:
            x = self.layer2(x)
            x = self.bn2(x)
        x = self.drop(x)
        return x


class DiagWinAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        window_size: Tuple[int] = (8, 8),  # todo: something about tuple window size
        num_heads: int = 6,
        attn_drop: float = 0.05,
        proj_drop: float = 0.05,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.num_patches = np.prod(window_size).item()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scale = (embed_dim // num_heads) ** -0.5

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

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor
    ):
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
        query = einops.rearrange(attn @ value + query, "nw nh (ws1 ws2) ch -> nw ws1 ws2 (nh ch)",
                                 ws1=self.window_size[0])
        query = self.norm(query)
        query = self.proj(query)
        query = self.proj_drop(query)
        return query


class BlindSpotBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int = 96,
        window_size: int = 8,
        shift_size: int = 0,
        num_heads: int = 6,
        input_size: Tuple[int] = (128, 128),
        attn_drop: float = 0.05,
        proj_drop: float = 0.05,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = Mlp(embed_dim, embed_dim, embed_dim)
        self.attn = DiagWinAttention(embed_dim, to_2tuple(window_size), num_heads, attn_drop, proj_drop)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.window_size = window_size
        self.shift_size = shift_size
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.attn_mask = self.calculate_mask(input_size)

    def shift_image(self, x: Optional[Tensor]):
        if x.shape[-1] != self.embed_dim or self.shift_size == 0:
            return x
        else:
            return torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

    def shift_image_reversed(self, x: Optional[Tensor]):
        if self.shift_size == 0:
            return x
        return torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

    def window_partition(self, x: Optional[Tensor]):
        if len(x.shape) == 3:
            return x
        return einops.rearrange(x, 'b (h wh) (w ww) c -> (b h w) (wh ww) c', wh=self.window_size, ww=self.window_size)

    def window_partition_reversed(self, x: Optional[Tensor], x_size: Iterable[int]):
        height, width = x_size
        h, w = height // self.window_size, width // self.window_size
        return einops.rearrange(x, '(b h w) wh ww c -> b (h wh) (w ww) c', h=h, w=w)

    def calculate_mask(self, x_size):
        attn_mask = torch.zeros((1, *x_size, 1))
        if self.shift_size != 0:
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[:, h, w, :] = cnt
                    cnt += 1
        attn_mask = self.window_partition(attn_mask)
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
        query, key, value = map(self.window_partition, map(self.shift_image, (query, key, value)))
        mask = self.attn_mask if image_size == self.input_size else self.calculate_mask(image_size).to(key.device)
        query = self.attn(query, key, value, mask=mask)
        query = self.shift_image_reversed(self.window_partition_reversed(query, image_size))
        query = query + self.mlp(self.norm2(query))
        return query


class SwinIA(nn.Module):

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 96,
        window_size: int = 8,
        num_heads: Tuple[int] = (6, 6)
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 1)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 1)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, window_size ** 2, embed_dim // num_heads[0]))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        self.blocks = nn.ModuleList([
            BlindSpotBlock(
                embed_dim=embed_dim,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                num_heads=n
            ) for i, n in enumerate(num_heads)
        ])

    def forward(self, x):
        x = self.conv_first(x)
        x = einops.rearrange(x, 'b c ... -> b ... c')
        q, k, v = self.absolute_pos_embed, x, x
        for block in self.blocks:
            q = block(q, k, v)
        q = einops.rearrange(q, 'b ... c -> b c ...')
        q = self.conv_last(q)
        return q
