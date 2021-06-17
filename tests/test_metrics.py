import unittest

import numpy as np
from scipy.misc import ascent
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from noise2same.util import normalize_min_mse


class MetricsTestCase(unittest.TestCase):
    def test_mse(self):
        y, x1, x2 = self._get_test_images()
        mse1 = mean_squared_error(*normalize_min_mse(y, x1))
        mse2 = mean_squared_error(*normalize_min_mse(y, x2))

        self.assertAlmostEqual(mse1, mse2, delta=1e-6)

    def test_psnr(self):
        y, x1, x2 = self._get_test_images()
        psnr1 = peak_signal_noise_ratio(*normalize_min_mse(y, x1))
        psnr2 = peak_signal_noise_ratio(*normalize_min_mse(y, x2))

        self.assertAlmostEqual(psnr1, psnr2, delta=1e-6)

    def test_ssim(self):
        y, x1, x2 = self._get_test_images()
        ssim1 = structural_similarity(*normalize_min_mse(y, x1))
        ssim2 = structural_similarity(*normalize_min_mse(y, x2))

        self.assertAlmostEqual(ssim1, ssim2, delta=1e-6)

    def _get_test_images(self):
        # ground truth image
        y = ascent().astype(np.float32)
        # input image to compare to
        x1 = y + 30 * np.random.normal(0, 1, y.shape)
        # a scaled and shifted version of x1
        x2 = 2 * x1 + 100

        return y, x1, x2


if __name__ == "__main__":
    unittest.main()
