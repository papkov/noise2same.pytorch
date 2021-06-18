import unittest

import numpy as np
from albumentations import PadIfNeeded

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


if __name__ == "__main__":
    unittest.main()
