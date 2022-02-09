from dataclasses import dataclass
from typing import Optional

import numpy as np

from noise2same.dataset.abc import AbstractNoiseDataset3DLarge


@dataclass
class DummyDataset3DLarge(AbstractNoiseDataset3DLarge):
    image: Optional[np.ndarray] = None
    image_size: int = 256
    path: str = "."

    def _read_large_image(self):
        if self.image is None:
            shape = (self.image_size,) * self.n_dim
            self.image = np.random.rand(*shape)
        self.image = self.image.astype(np.float32)
