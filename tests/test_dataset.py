import unittest

import numpy as np
from albumentations import PadIfNeeded

from noise2same.dataset.util import mask_like_image
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


if __name__ == "__main__":
    unittest.main()
