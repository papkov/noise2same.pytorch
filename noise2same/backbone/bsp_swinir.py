# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------
import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .swinir import (
    window_partition,
    window_reverse,
    SwinTransformerBlock,
    BasicLayer,
    RSTB,
    SwinIR
)


class BSpSwinTransformerBlock(SwinTransformerBlock):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, first=False):
        super().__init__(
            dim, input_resolution, num_heads, window_size=window_size, shift_size=shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer
        )
        self.first = first

    def forward(self, x, x_size, mask=None):
        shortcut = x
        batch_size = x.shape[0]
        x = self.norm1(x)
        x = einops.rearrange(x, "b (h w) c -> b h w c", h=x_size[0])

        shifted_mask = mask
        shifted_x = x

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = einops.rearrange(x_windows, "nw ... c -> nw (...) c")

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_mask = self.attn_mask
        else:
            attn_mask = self.calculate_mask(x_size).to(x.device)

        # replicating the mask for the whole batch
        if attn_mask is not None:
            attn_mask = einops.repeat(attn_mask, "... -> b ...", b=batch_size)

        # adding input mask to existing blind-spot mask
        if self.first:
            bsp_mask = window_partition(einops.rearrange(shifted_mask, "b 1 ... -> b ... 1"), self.window_size)
            bsp_mask = einops.rearrange(bsp_mask, "(b nw) ... -> b nw (...)", b=batch_size)
            bsp_mask = einops.repeat(bsp_mask, "... n -> ... repeat n", repeat=bsp_mask.shape[-1])
            if attn_mask is None:
                attn_mask = bsp_mask
            attn_mask = attn_mask.masked_fill(bsp_mask != 0, float(-10 ** 9))
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = einops.rearrange(attn_windows, "... (ws1 ws2) c -> ... ws1 ws2 c", ws1=self.window_size)
        shifted_x = window_reverse(attn_windows, self.window_size, *x_size)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = einops.rearrange(x, "b ... c -> b (...) c")

        # FFN
        x = self.drop_path(x) if self.first else shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BSpBasicLayer(BasicLayer):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, first=False):

        super().__init__(
            dim, input_resolution, depth, num_heads, window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint
        )

        # build blocks
        self.blocks = nn.ModuleList([
            BSpSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,
                                    first=first and i == 0)
            for i in range(depth)])

    def forward(self, x, x_size, mask=None):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, mask=mask)
            else:
                x = blk(x, x_size, mask=mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BSpRSTB(RSTB):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, resi_connection='1conv', first=False):
        super().__init__(
            dim, input_resolution, depth, num_heads, window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint,
            img_size=img_size, resi_connection=resi_connection
        )
        self.first = first

        self.residual_group = BSpBasicLayer(dim=dim,
                                            input_resolution=input_resolution,
                                            depth=depth,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            drop=drop, attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            norm_layer=norm_layer,
                                            downsample=downsample,
                                            use_checkpoint=use_checkpoint,
                                            first=first)

    def forward(self, x, x_size, mask=None):
        return self.patch_embed(
            self.conv(
                self.patch_unembed(
                    self.residual_group(x, x_size, mask=mask),
                    x_size)
            )
        ) + (0 if self.first else x)


class BSpSwinIR(SwinIR):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, in_chans=3, embed_dim=96,
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., resi_connection='1conv',
                 **kwargs):
        super().__init__(
            img_size=img_size, in_chans=in_chans, embed_dim=embed_dim,
            depths=depths, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, upscale=upscale, img_range=img_range, resi_connection=resi_connection,
            **kwargs
        )

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 1)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BSpRSTB(dim=embed_dim,
                            input_resolution=(self.patches_resolution[0],
                                              self.patches_resolution[1]),
                            depth=depths[i_layer],
                            num_heads=num_heads[i_layer],
                            window_size=window_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                            # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            resi_connection=resi_connection,
                            first=i_layer == 0)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def forward_features(self, x, mask=None):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, x_size, mask=mask)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, mask=None):
        x = self.check_image_size(x)

        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first, mask=mask))
        x = self.conv_last(res)

        return x
