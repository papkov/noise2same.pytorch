import unittest

import numpy as np
import torch
from albumentations import PadIfNeeded

from noise2same.dataset.util import mask_like_image, PadAndCropResizer
from noise2same.util import crop_as


class TestDataset(unittest.TestCase):
    def test_crop_as(self):
        for divisor in (2, 4, 8, 16, 32, 64):
            pad = PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=divisor,
                pad_width_divisor=divisor,
            )

            image = np.random.uniform(size=(180, 180, 1))
            padded = pad(image=image)["image"]
            cropped = crop_as(padded, image)
            print(padded.shape, cropped.shape)
            self.assertEqual(cropped.shape, image.shape)
            self.assertTrue(np.all(cropped == image))

    def test_mask_2d(self, mask_percentage: float = 0.5):
        shape = (64, 64, 3)
        img = np.zeros(shape)
        mask = mask_like_image(img, mask_percentage=mask_percentage, channels_last=True)
        result = mask.mean() * 100
        self.assertAlmostEqual(mask_percentage, result, delta=0.1)

    def test_mask_3d(self, mask_percentage: float = 0.5):
        shape = (1, 16, 64, 64)
        img = np.zeros(shape)
        mask = mask_like_image(
            img, mask_percentage=mask_percentage, channels_last=False
        )
        result = mask.mean() * 100
        self.assertAlmostEqual(mask_percentage, result, delta=0.1)


class TestResizer(unittest.TestCase):

    def test_resize_square_odd(self):
        resizer = PadAndCropResizer(div_n=2, square=True)
        tensor = torch.rand(1, 3, 63, 67)
        resized = resizer.before(tensor, exclude=(0, 1))
        self.assertEqual(resized.shape, (1, 3, 68, 68))

        cropped = resizer.after(resized)
        self.assertEqual(cropped.shape, tensor.shape)

    def test_resize_square_even(self):
        resizer = PadAndCropResizer(div_n=8, square=True)
        tensor = torch.rand(1, 1, 328, 488)
        resized = resizer.before(tensor, exclude=(0, 1))
        self.assertEqual(resized.shape, (1, 1, 488, 488))

        cropped = resizer.after(resized)
        self.assertEqual(cropped.shape, tensor.shape)

    def test_resize_square_odd_numpy(self):
        resizer = PadAndCropResizer(div_n=2, square=True)
        tensor = np.random.uniform(size=(3, 63, 67))
        resized = resizer.before(tensor, exclude=0)
        self.assertEqual(resized.shape, (3, 68, 68))

        cropped = resizer.after(resized)
        self.assertEqual(cropped.shape, tensor.shape)

    def test_resize_square_even_numpy(self):
        resizer = PadAndCropResizer(div_n=8, square=True)
        tensor = np.random.uniform(size=(1, 328, 488))
        resized = resizer.before(tensor, exclude=0)
        self.assertEqual(resized.shape, (1, 488, 488))

        cropped = resizer.after(resized)
        self.assertEqual(cropped.shape, tensor.shape)


if __name__ == "__main__":
    unittest.main()
