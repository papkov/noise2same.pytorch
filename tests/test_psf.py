import unittest

import numpy as np
import torch

from noise2same.psf.psf_convolution import PSF, PSFParameter


class PSFTestCase(unittest.TestCase):
    def test_psf_fft(self, s: int = 7):
        kernel = np.random.rand(s, s, s)
        patch = torch.rand(1, 1, 64, 64, 64)

        psf = PSF(kernel, fft=False)
        psf_fft = PSF(kernel, fft=True)

        psf_out = psf(patch)
        psf_fft_out = psf_fft(patch)

        self.assertTrue(patch.shape == psf_out.shape)
        self.assertTrue(torch.allclose(psf_out, psf_fft_out))

    def test_psf_delta(self, s: int = 7):
        kernel = np.zeros((s, s, s))
        kernel[s // 2, s // 2, s // 2] = 1  # delta function

        patch = torch.rand(1, 1, 64, 64, 64)

        psf = PSF(kernel, fft=False)
        psf_out = psf(patch)

        self.assertTrue(torch.allclose(psf_out, patch))

    def test_psf_fft_delta(self, s: int = 7):
        kernel = np.zeros((s, s, s))
        kernel[s // 2, s // 2, s // 2] = 1  # delta function

        patch = torch.rand(1, 1, 64, 64, 64)

        psf = PSF(kernel, fft=True)
        psf_out = psf(patch)

        # 1e-7 does not work for some reason
        self.assertTrue(torch.allclose(psf_out, patch, atol=1e-6))

    def test_psf_parameter(self):
        kernel = np.random.rand(7, 7, 7)
        patch = torch.rand(1, 1, 64, 64, 64)

        psf = PSF(kernel, fft=True)
        psf_param = PSFParameter(kernel, fft=True)

        psf_out = psf(patch)
        psf_param_out = psf_param(patch)

        self.assertTrue(patch.shape == psf_out.shape)
        self.assertTrue(torch.allclose(psf_out, psf_param_out))

    def test_large_psf(self):
        kernel = np.random.rand(128, 256, 512)
        patch = torch.rand(1, 1, 128, 128, 128)

        psf = PSFParameter(kernel, fft=True)
        psf_out = psf(patch)

        self.assertTrue(patch.shape == psf_out.shape, f"Output shape: {psf_out.shape}")

    def test_psf_auto_padding(self):
        kernel = np.random.rand(7, 7, 7)
        patch = torch.rand(1, 1, 64, 64, 64)

        psf = PSFParameter(kernel, fft=True, auto_padding=False)
        psf_auto = PSFParameter(kernel, fft=True, auto_padding=True)

        psf_out = psf(patch)
        psf_auto_out = psf_auto(patch)

        self.assertTrue(patch.shape == psf_auto_out.shape)
        self.assertTrue(torch.allclose(psf_out, psf_auto_out))


if __name__ == "__main__":
    unittest.main()
