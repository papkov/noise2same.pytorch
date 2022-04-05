import unittest
from functools import partial

import torch

from noise2same.ffc import FFC, FFCInc


class FFCTestCase(unittest.TestCase):
    def test_inception_ffc_identity(self):
        x = (torch.rand(1, 16, 64, 64), torch.rand(1, 16, 64, 64))
        params = dict(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            ratio_gin=0.5,
            ratio_gout=0.5,
            padding=1,
        )

        ffc = FFC(**params)
        ffc_inc = FFCInc(**params, ratio_ffc=1)

        for name, params in ffc.named_parameters():
            ffc_inc.state_dict()[name].copy_(params)

        out_ffc_l, out_ffc_g = ffc(x)
        out_ffc_inc_l, out_ffc_inc_g = ffc_inc(x)

        self.assertTrue(torch.allclose(out_ffc_l, out_ffc_inc_l))
        self.assertTrue(torch.allclose(out_ffc_g, out_ffc_inc_g))

    def test_inception_ffc_forward(self):
        x = (torch.rand(1, 16, 64, 64), torch.rand(1, 16, 64, 64))
        params = dict(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            ratio_gin=0.5,
            ratio_gout=0.5,
            padding=1,
        )

        ffc = FFC(**params)
        ffc_inc = FFCInc(**params, ratio_ffc=0.5)

        out_ffc_l, out_ffc_g = ffc(x)
        out_ffc_inc_l, out_ffc_inc_g = ffc_inc(x)

        self.assertFalse(torch.allclose(out_ffc_l, out_ffc_inc_l))
        self.assertFalse(torch.allclose(out_ffc_g, out_ffc_inc_g))


if __name__ == "__main__":
    unittest.main()
