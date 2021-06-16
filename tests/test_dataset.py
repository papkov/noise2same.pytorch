import unittest
from noise2same.util import crop_as
from albumentations import PadIfNeeded
import numpy as np


class TestDataset(unittest.TestCase):
    def test_crop_as(self):
        pad = PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=32,
                    pad_width_divisor=32,
                )

        image = np.random.uniform(size=(180, 180, 1))
        padded = pad(image=image)["image"]
        cropped = crop_as(padded, image)
        self.assertEqual(cropped.shape, image.shape)
        self.assertTrue(np.all(cropped == image))


if __name__ == '__main__':
    unittest.main()
