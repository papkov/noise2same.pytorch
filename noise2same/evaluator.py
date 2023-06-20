import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import TileMerger
from torch import Tensor as T
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.util import PadAndCropResizer
from noise2same.denoiser import Denoiser
from noise2same.backbone.unet import UNet
from noise2same.backbone.swinir import SwinIR
from noise2same.dataset.tiling import TiledImageFactory
from noise2same.util import crop_as, calculate_scores, detach_to_np


class Evaluator(object):
    def __init__(
            self,
            denoiser: Denoiser,
            device: str = "cuda",
            checkpoint_path: Optional[str] = None,
    ):
        """
        Model evaluator, describes inference for different data formats
        :param denoiser: model architecture to evaluate
        :param device: str, device to run inference
        :param checkpoint_path: optional str, path to the model checkpoint
        """
        self.model = denoiser
        self.device = device
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        self.model.to(device)

        if isinstance(self.model.backbone, UNet):
            self.resizer = PadAndCropResizer(
                mode="reflect", div_n=2 ** self.model.backbone.depth
            )
        elif isinstance(self.model.backbone, SwinIR):
            self.resizer = PadAndCropResizer(
                mode="reflect", div_n=self.model.backbone.window_size
            )
        else:
            self.resizer = PadAndCropResizer(div_n=1)

    @torch.no_grad()
    def inference(
        self,
        loader: DataLoader,
        half: bool = False,
        empty_cache: bool = False,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        Run inference for a given dataloader
        :param loader: DataLoader
        :param half: bool, if use half precision
        :param empty_cache: bool, if empty CUDA cache after each iteration
        :return: List[Dict[key, output]]
        """
        self.model.eval()

        outputs = []
        iterator = tqdm(loader, desc="inference", position=0, leave=True)
        times = []
        errors_num = 0
        indices = []
        for i, batch in enumerate(iterator):
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                start = time.time()
                with autocast(enabled=half):
                    x_out = self.model.forward(batch["image"])

                x_out = detach_to_np(x_out, batch["mean"], batch["std"])

                end = time.time()
                times.append(end - start)

                outputs.append(x_out)
                iterator.set_postfix(
                    {
                        "shape": batch["image"].shape,
                        "reserved": torch.cuda.memory_reserved(0) / (1024 ** 2),
                        "allocated": torch.cuda.memory_allocated(0) / (1024 ** 2),
                    }
                )

                if empty_cache:
                    torch.cuda.empty_cache()
            except RuntimeError:
                errors_num += 1
                print('Skipping image ', i)
                pass
            else:
                indices.append(i)

        print(f"Average inference time: {np.mean(times) * 1000:.2f} ms")
        print(f'Dropped images rate: {errors_num / len(loader)}')
        # TODO standardize output format
        return outputs, indices  # СТЫД

    @torch.no_grad()
    def inference_single_image_dataset(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            num_workers: int = 0,
            crop_border: int = 0,
            device: str = "cpu",
            half: bool = False,
            empty_cache: bool = False,
            key: str = "image",
    ) -> Dict[str, np.ndarray]:
        """
        Run inference for a single image represented as Dataset
        Here, we assume that dataset was tiled and has a `tiler` attribute

        :param dataset: Dataset representing a single large tiled image
        :param batch_size: int, batch size for DataLoader
        :param num_workers: int, number of workers for DataLoader
        :param crop_border: int, border pixels to crop when merging tiles
        :param device: str, device where to accumulate merging tiles
        :param half: bool, if use half precision
        :param empty_cache: bool, if empty CUDA cache after
        :param key: str, which output key to accumulate
        :return: numpy array, merged image
        """
        assert hasattr(dataset, "tiler"), "Dataset should have a `tiler` attribute"

        self.model.eval()

        merger = TileMerger(
            image_shape=dataset.tiler.target_shape,
            channels=self.model.backbone.in_channels,
            weight=dataset.tiler.weight,
            device=device,
            crop_border=crop_border,
            default_value=0,
        )
        # print(f'Created merger for image {merger.image.shape}')

        iterator = dataset
        if batch_size > 1:
            iterator = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            )
        iterator = tqdm(iterator, desc="Predict")

        for i, batch in enumerate(iterator):
            # We can iterate over dataset, hence need to unsqueeze batch dim
            if batch_size == 1:
                batch["image"] = batch["image"][None, ...]
                batch["crop"] = batch["crop"][None, ...]

            # We don't need move to device for `crop`
            batch = {
                k: v.to(self.device) if k != "crop" else v for k, v in batch.items()
            }
            with autocast(enabled=half):
                pred_batch = (
                        self.model.forward(batch["image"])[key]
                        * batch["std"]
                        + batch["mean"]
                )
            iterator.set_postfix(
                {
                    "in_shape": tuple(batch["image"].shape),
                    "out_shape": tuple(pred_batch.shape),
                    "crop": batch["crop"],
                }
            )

            try:
                merger.integrate_batch(batch=pred_batch, crop_coords=batch["crop"])
            except TypeError as e:
                raise TypeError(
                    f"Error on batch {i} with shape {batch['image'].shape}, crop {batch['crop']}"
                ) from e
            if empty_cache:
                torch.cuda.empty_cache()

        merger.merge_()
        return {'image': dataset.tiler.crop_to_original_size(
            merger.image["image"].cpu().numpy()[0]
        )}

    @torch.no_grad()
    def inference_single_image_tensor(
        self,
        image: torch.Tensor,
        standardize: bool = True,
        im_mean: Optional[float] = None,
        im_std: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Run inference for a single image represented as Dataset

        :param image: torch.Tensor
        :param standardize: bool, if subtract mean and divide by std
        :param im_mean: float, precalculated image mean
        :param im_std: float, precalculated image std
        :return: torch.Tensor
        """

        if not standardize:
            im_mean, im_std = 0, 1

        if im_mean is None:
            im_mean = image.mean()

        if im_std is None:
            im_std = image.std()

        image = (image - im_mean) / im_std

        image = self.resizer.before(image, exclude=0)[None, ...]
        out = self.model(image.to(self.device)).detach().cpu()
        out = self.resizer.after(out[0], exclude=0)
        out = out * im_std + im_mean

        return out

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            checkpoint["model"].pop("mask_kernel.kernel", None)
            checkpoint = checkpoint["model"]
        self.model.load_state_dict(checkpoint, strict=False)

    @torch.no_grad()
    def evaluate(
            self,
            dataset: AbstractNoiseDataset,
            factory: Optional[TiledImageFactory] = None,
            half: bool = False,
            empty_cache: bool = False,
            key: str = 'image',
            keep_images: bool = False,
            metrics: Tuple[str, ...] = ("rmse", "psnr", "ssim"),
            num_workers: int = 0,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run inference for a given dataloader
        :param dataset: AbstractNoiseDataset
        :param factory: Optional[TiledImageFactory]
        :param half: bool, if use half precision
        :param empty_cache: bool, if empty CUDA cache after each iteration
        :param key: str, key to use for the output [image, deconv]
        :param keep_images: bool, if add prediction arrays to result
        :param metrics: tuple of metrics to calculate
        :param num_workers: int, number of workers for DataLoader
        :return: List[Dict[key, output]]
        """
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        self.model.eval()
        outputs = []
        iterator = tqdm(loader, desc="inference", position=0, leave=True)
        full_inference_time, test_size = 0, 0
        if factory is None:
            inference = partial(self._inference_batch, half=half, empty_cache=empty_cache, key=key)
        else:
            inference = partial(self._inference_large_batch, factory=factory, half=half,
                                empty_cache=empty_cache, key=key)
        for i, batch in enumerate(iterator):
            out, inference_time = inference(batch)
            out = self._revert_batch(out, ['image', 'ground_truth'])
            for j, (pred, gt) in enumerate(zip(out['image'], out['ground_truth'])):
                if dataset.n_dim == 2:
                    if dataset.data_range == 1:
                        pred = pred * 255
                        gt = gt * 255
                    pred = np.clip(pred + 0.5, 0, 255).astype(np.uint8).astype(np.float32)
                    gt = np.clip(gt + 0.5, 0, 255).astype(np.uint8).astype(np.float32)
                scores = calculate_scores(
                    pred, gt,
                    multichannel=True,
                    data_range=255.0 if dataset.n_dim > 2 else 1.0,
                    normalize_pairs=dataset.n_dim > 2,
                    metrics=metrics,
                )
                if keep_images:
                    scores['image'] = pred
                outputs.append(scores)
            full_inference_time += inference_time
            test_size += batch['image'].shape[0]

        print(f"Average inference time: {full_inference_time / test_size * 1000:.2f} ms")
        return outputs

    def _inference_large_batch(
        self,
        batch: Dict[str, T],
        factory: TiledImageFactory,
        half: bool = False,
        empty_cache: bool = False,
        key: str = 'image',
    ) -> Tuple[Dict[str, np.ndarray], float]:
        output = {'image': []}
        full_inference_time = 0
        for i in range(batch['image'].shape[0]):
            image = {k: v[i] for k, v in batch.items()}
            loader, merger = factory.produce(image)
            iterator = tqdm(loader, desc="Predict")
            for tile_batch in iterator:
                pred_batch, inference_time = self._inference_batch(tile_batch, half=half, empty_cache=empty_cache,
                                                                   key=key)
                iterator.set_postfix(
                    {
                        "in_shape": tuple(tile_batch["image"].shape),
                        "out_shape": tuple(pred_batch['image'].shape),
                        "crop": tile_batch["crop"],
                    }
                )

                merger.integrate_batch(batch=pred_batch['image'], crop_coords=tile_batch["crop"])
                full_inference_time += inference_time

            merger.merge_()
            output['image'].append(loader.dataset.slicer.crop_to_original_size(
                np.moveaxis(merger.image["image"].cpu().numpy(), 0, -1)
            ))
        batch['image'] = torch.from_numpy(np.moveaxis(np.array(output['image']), -1, 1))
        return batch, full_inference_time

    def _inference_batch(
        self,
        batch: Dict[str, T],
        half: bool = False,
        empty_cache: bool = False,
        key: str = 'image',
    ) -> Tuple[Dict[str, T], float]:
        batch = {k: v.to(self.device) if k in ['image', 'ground_truth', 'std', 'mean'] else v for k, v in batch.items()}
        start = time.time()
        with autocast(enabled=half):
            try:
                batch['image'] = self.model.forward(batch['image'])[key]
            except RuntimeError as e:
                raise RuntimeError(f"Error during inference on batch {batch['image'].shape}, "
                                   f"half={half}, empty_cache={empty_cache}, key={key}") from e
        end = time.time()
        if empty_cache:
            torch.cuda.empty_cache()
        return batch, end - start

    def _revert_batch(
            self,
            batch: Dict[str, T],
            keys: List[str]
    ) -> Dict[str, np.ndarray]:
        out = detach_to_np({k: batch[k] for k in keys}, batch['mean'], batch['std'])
        for k, v in out.items():
            out[k] = np.array([crop_as(im, sh) for im, sh in zip(v, batch['shape'])])
        return out
