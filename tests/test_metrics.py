import unittest

import numpy as np
from scipy.misc import ascent
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

from noise2same.util import normalize_min_mse


def test_mse():
    y, x1, x2 = get_test_images()
    mse1 = mean_squared_error(y, normalize_min_mse(y, x1))
    mse2 = mean_squared_error(y, normalize_min_mse(y, x2))

    assert np.isclose(mse1, mse2, atol=1e-6)


def test_psnr():
    y, x1, x2 = get_test_images()
    psnr1 = peak_signal_noise_ratio(y, normalize_min_mse(y, x1), data_range=255)
    psnr2 = peak_signal_noise_ratio(y, normalize_min_mse(y, x2), data_range=255)

    assert np.isclose(psnr1, psnr2, atol=1e-6)


def test_ssim():
    y, x1, x2 = get_test_images()
    ssim1 = structural_similarity(y, normalize_min_mse(y, x1), data_range=255)
    ssim2 = structural_similarity(y, normalize_min_mse(y, x2), data_range=255)

    assert np.isclose(ssim1, ssim2, atol=1e-6)


def get_test_images():
    # ground truth image
    y = ascent().astype(np.float32)
    # input image to compare to
    x1 = y + 30 * np.random.normal(0, 1, y.shape)
    # a scaled and shifted version of x1
    x2 = 2 * x1 + 100

    return y, x1, x2
