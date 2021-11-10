from typing import Optional, Tuple

import torch
from einops import rearrange, reduce
from torch import Tensor as T
from torch.nn.modules.loss import _Loss


class PixelContrastLoss(_Loss):
    def __init__(self, temperature: float = 0.1):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature

    def forward(self, out_raw: T, out_mask: T, mask: T) -> T:
        """

        :param out_raw: tensor (B, E, H, W)
        :param out_mask: tensor (B, E, H, W)
        :param mask:  tensor (B, 1, H, W)
        :return:
        """
        mask = rearrange(mask, "b e h w -> (b e h w)")  # e == 1

        emb_raw = rearrange(out_raw, "b e h w -> (b h w) e")[mask.bool()]
        emb_mask = rearrange(out_mask, "b e h w -> (b h w) e")[mask.bool()]
        rand_idx = torch.randperm(emb_raw.size(0))

        pos_dot = torch.einsum("be,be->b", emb_raw, emb_mask) / self.temperature
        neg_dot_raw = (
            torch.einsum("be,be->b", emb_raw[rand_idx], emb_mask) / self.temperature
        )
        neg_dot_mask = (
            torch.einsum("be,be->b", emb_raw, emb_mask[rand_idx]) / self.temperature
        )
        neg_dot = torch.stack([neg_dot_raw, neg_dot_mask], dim=-1)

        pos_max_val = torch.max(pos_dot)
        neg_max_val = torch.max(neg_dot)
        max_val = torch.max(torch.stack([pos_max_val, neg_max_val]))

        numerator = torch.exp(pos_dot - max_val)

        denominator = (
            reduce(torch.exp(neg_dot - max_val), "b k -> b", "sum") + numerator
        )
        loss = -torch.log((numerator / denominator) + 1e-8)
        return loss
