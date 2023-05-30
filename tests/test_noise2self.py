import unittest

import numpy as np
import torch

from noise2same import denoiser
from noise2same.backbone import unet
from noise2same.dataset.util import mask_like_image
from noise2same.psf.psf_convolution import PSFParameter


class TestNoise2Self(unittest.TestCase):
    def test_masked_loss(self):
        dnsr = denoiser.Noise2Self(
            backbone=unet.UNet(in_channels=1, base_channels=4),
            head=unet.RegressionHead(in_channels=4, out_channels=1),
        )

        image = np.random.uniform(size=(64, 64, 1)).astype(np.float32)
        mask_1 = mask_like_image(image, mask_percentage=0.5)
        mask_2 = mask_like_image(image, mask_percentage=0.5)

        x = torch.from_numpy(np.rollaxis(image, -1, 0)).float()[None, ...]
        mask_1 = torch.from_numpy(np.rollaxis(mask_1, -1, 0)).float()[None, ...]
        mask_2 = torch.from_numpy(np.rollaxis(mask_2, -1, 0)).float()[None, ...]

        out_1 = dnsr(x, mask_1)
        out_2 = dnsr(x, mask_2)

        loss_1, loss_dict_1 = dnsr.compute_loss({'image': x, 'mask': mask_1}, out_1)
        loss_2, loss_dict_2 = dnsr.compute_loss({'image': x, 'mask': mask_2}, out_2)

        self.assertNotEqual(loss_dict_1['loss'], loss_dict_2['loss'])

    def test_masked_loss_same(self):
        dnsr = denoiser.Noise2Self(
            masking='donut',  # otherwise random noise in masking produces different loss
            backbone=unet.UNet(in_channels=1, base_channels=4),
            head=unet.RegressionHead(in_channels=4, out_channels=1),
        )

        image = np.random.uniform(size=(64, 64, 1)).astype(np.float32)
        mask = mask_like_image(image, mask_percentage=0.5)

        x = torch.from_numpy(np.rollaxis(image, -1, 0)).float()[None, ...]
        mask_1 = torch.from_numpy(np.rollaxis(mask, -1, 0)).float()[None, ...]
        mask_2 = torch.from_numpy(np.rollaxis(mask, -1, 0)).float()[None, ...]

        out_1 = dnsr(x, mask_1)
        out_2 = dnsr(x, mask_2)

        loss_1, loss_dict_1 = dnsr.compute_loss({'image': x, 'mask': mask_1}, out_1)
        loss_2, loss_dict_2 = dnsr.compute_loss({'image': x, 'mask': mask_2}, out_2)

        self.assertEqual(loss_dict_1['loss'], loss_dict_2['loss'])

    def test_deconvolution(self):
        psf = PSFParameter(np.ones((3, 3)) / 9)
        dnsr = denoiser.Noise2SelfDeconvolution(
            backbone=unet.UNet(in_channels=1, base_channels=4),
            head=unet.RegressionHead(in_channels=4, out_channels=1),
            psf=psf,
            lambda_bound=0.1,
            lambda_sharp=0.1,
            regularization_key='image',
        )

        image = np.random.uniform(size=(64, 64, 1)).astype(np.float32)
        mask = mask_like_image(image, mask_percentage=0.5)

        x = torch.from_numpy(np.rollaxis(image, -1, 0)).float()[None, ...]
        mask = torch.from_numpy(np.rollaxis(mask, -1, 0)).float()[None, ...]

        out = dnsr(x, mask)

        for k in ['image', 'image/deconv']:
            self.assertIn(k, out)

        loss, loss_dict = dnsr.compute_loss({'image': x, 'mask': mask, 'mean': 0, 'std': 1}, out)

        for k in ['loss', 'bound_loss', 'sharp_loss']:
            self.assertIn(k, loss_dict)



if __name__ == '__main__':
    unittest.main()