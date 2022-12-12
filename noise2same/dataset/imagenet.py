from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
from tqdm.auto import tqdm

from noise2same.dataset.abc import AbstractNoiseDataset2D
from noise2same.dataset.util import (
    add_microscope_blur_2d,
    add_poisson_gaussian_noise,
    get_normalized_psf,
    normalize,
)


@dataclass
class ImagenetDatasetPrepared(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/ImageNet"
    mode: str = "train"
    version: int = 0  # two noisy copies exist (0, 1)
    standardize_by_channel: bool = True
    add_blur_and_noise: bool = False

    def _validate(self) -> bool:
        assert self.mode in ("train", "val")
        assert self.version in (0, 1)
        return True

    def _get_images(self) -> Union[List[str], np.ndarray]:
        if self.add_blur_and_noise:
            if (self.path / f"{self.mode}_blurred.npy").exists():
                print("Loading blurred images from file")
                self.psf = np.load(self.path / f"psf.npy")
                return np.load(self.path / f"{self.mode}_blurred.npy")
            else:
                print("Creating blurred images")
                # [0] is the original image, [1, 2] is two noisy copies of image
                images = np.load(self.path / f"{self.mode}.npy")[:, 0].astype(
                    np.float32
                )
                for i, image_clipped in enumerate(
                    tqdm(images, desc="Adding blur and noise")
                ):
                    image_clipped = normalize(image_clipped)
                    blurred_image, self.psf = add_microscope_blur_2d(
                        image_clipped, multi_channel=True
                    )
                    noisy_blurred_image = add_poisson_gaussian_noise(
                        blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10
                    )
                    assert not np.allclose(noisy_blurred_image, 0)
                    images[i] = noisy_blurred_image
                np.save(self.path / f"{self.mode}_blurred.npy", images)
                np.save(self.path / f"psf.npy", self.psf)
                return images
        return np.load(self.path / f"{self.mode}.npy")[:, self.version + 1]

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        return image_or_path


@dataclass
class ImagenetDatasetTest(AbstractNoiseDataset2D):
    path: Union[Path, str] = "data/ImageNet"
    standardize_by_channel: bool = True
    add_blur_and_noise: bool = False

    def _get_images(self) -> Union[List[str], np.ndarray]:
        if self.add_blur_and_noise:
            psf_path = self.path / "psf.npy"
            if psf_path.exists():
                self.psf = np.load(psf_path)
            else:
                self.psf = get_normalized_psf()
        return sorted((self.path / "test").glob("*.npy"))

    def _read_image(self, image_or_path: Union[str, np.ndarray]) -> np.ndarray:
        if self.add_blur_and_noise:
            blurred_path = str(image_or_path).replace("test", "test_blurred")
            if Path(blurred_path).exists():
                return np.load(blurred_path)
            else:
                image_clipped = normalize(np.load(image_or_path)[0].astype(np.float32))
                blurred_image, self.psf = add_microscope_blur_2d(
                    image_clipped, multi_channel=True
                )
                noisy_blurred_image = add_poisson_gaussian_noise(
                    blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10
                )
                Path(blurred_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(blurred_path, noisy_blurred_image)
                return noisy_blurred_image
        return np.load(image_or_path)[1]
