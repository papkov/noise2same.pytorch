import unittest

import torch
from einops import rearrange

from noise2same.contrast import PixelContrastLoss

class MyTestCase(unittest.TestCase):
    def test_masking(self, bs: int = 2, n_emb: int = 8, h: int = 4, w: int = 4):

        t = torch.randint(0, 10, (bs, n_emb, h, w))
        mask = torch.randn((bs, 1, h, w)).ge(0.5)
        mask[0, 0, 0, 0] = True
        mask[-1, -1, -1, -1] = True

        t_flat = rearrange(t, "b e h w -> (b h w) e", b=bs, h=h, w=w)
        mask = rearrange(mask, "b e h w -> (b h w e)", b=bs, h=h, w=w)
        print(t_flat.shape, mask.shape)
        t_masked = rearrange(t_flat[mask], "(b m) e -> b m e", b=bs)

        self.assertTrue(torch.all(torch.eq(t[0, :, 0, 0], t_masked[0, 0])))
        self.assertTrue(torch.all(torch.eq(t[-1, :, -1, -1], t_masked[-1, -1])))

    def test_loss(self, bs: int = 2, n_emb: int = 8, h: int = 4, w: int = 4):
        # mask = torch.randn(bs * h * w).ge(0.5)
        # out_raw = torch.randn(bs * n_emb * h * w)
        # out_mask = out_raw.clone()
        # out_mask[mask] = out_mask[mask] + torch.randn_like(out_mask)[mask] / 100
        #
        # out_raw = out_raw.reshape(bs, n_emb, h, w)
        # out_mask = out_mask.reshape(bs, n_emb, h, w)

        mask = torch.randn((bs, 1, h, w)).ge(0.5)
        out_raw = torch.randn((bs, n_emb, h, w))
        out_mask = torch.randn((bs, n_emb, h, w))

        loss = PixelContrastLoss()
        res = loss(out_raw, out_mask, mask)
        print(res.mean())




if __name__ == "__main__":
    unittest.main()
