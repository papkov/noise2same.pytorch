from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import TileMerger
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

from noise2same.dataset.util import PadAndCropResizer
from noise2same.model import Noise2Same
from noise2same.backbone.unet import UNet
from noise2same.backbone.swinir import SwinIR


class Evaluator(object):
    def __init__(
        self,
        model: Noise2Same,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        masked: bool = False,
    ):
        """
        Model evaluator, describes inference for different data formats
        :param model: model architecture to evaluate
        :param device: str, device to run inference
        :param checkpoint_path: optional str, path to the model checkpoint
        :param masked: if perform forward pass masked
        """

        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.masked = masked

        self.model.to(device)

        if isinstance(self.model.net, UNet):
            self.resizer = PadAndCropResizer(
                mode="reflect", div_n=2 ** self.model.net.depth
            )
        elif isinstance(self.model.net, SwinIR):
            self.resizer = PadAndCropResizer(
                mode="reflect", div_n=self.model.net.window_size
            )
        else:
            self.resizer = PadAndCropResizer(div_n=1)

    @torch.no_grad()
    def inference(
        self,
        loader: DataLoader,
        half: bool = False,
        empty_cache: bool = False,
        convolve: bool = False,
        key: str = "image",
    ) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
        """
        Run inference for a given dataloader
        :param loader: DataLoader
        :param half: bool, if use half precision
        :param empty_cache: bool, if empty CUDA cache after each iteration
        :param convolve: bool, if convolve the output with a PSF
        :param key: str, key to use for the output [image, deconv]
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
                    if self.masked:
                        # TODO remove randomness
                        # idea: use the same mask for all images? mask as tta?
                        out = self.model.forward_masked(
                            batch["image"], batch["mask"], convolve=convolve
                        )
                    else:
                        out = self.model.forward(batch["image"], convolve=convolve)
                    out_raw = out[key] * batch["std"] + batch["mean"]

                out_raw = {"image": np.moveaxis(out_raw.detach().cpu().numpy(), 1, -1)}
                if self.model.lambda_proj > 0:
                    out_raw.update(
                        {"proj": np.moveaxis(out["proj"].detach().cpu().numpy(), 1, -1)}
                    )

                end = time.time()
                times.append(end - start)

                outputs.append(out_raw)
                iterator.set_postfix(
                    {
                        "shape": out_raw["image"].shape,
                        "reserved": torch.cuda.memory_reserved(0) / (1024 ** 2),
                        "allocated": torch.cuda.memory_allocated(0) / (1024 ** 2),
                    }
                )

                if empty_cache:
                    torch.cuda.empty_cache()
            except RuntimeError:
                errors_num += 1
                pass
            else:
                indices.append(i)

        print(f"Average inference time: {np.mean(times) * 1000:.2f} ms")
        print(f'Dropped images rate: {errors_num / len(loader)}')
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
        convolve: bool = False,
    ) -> np.ndarray:
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
        :param convolve: bool, if convolve the output
        :return: numpy array, merged image
        """
        assert hasattr(dataset, "tiler"), "Dataset should have a `tiler` attribute"

        self.model.eval()

        merger = TileMerger(
            image_shape=dataset.tiler.target_shape,
            channels=self.model.in_channels,
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
                    self.model.forward(batch["image"], convolve=convolve)[key]
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

            merger.integrate_batch(batch=pred_batch, crop_coords=batch["crop"])
            if empty_cache:
                torch.cuda.empty_cache()

        merger.merge_()
        return dataset.tiler.crop_to_original_size(
            merger.image["image"].cpu().numpy()[0]
        )

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
