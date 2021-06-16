from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from torch import Tensor as T
from torch import nn
from torch.nn.functional import conv2d, conv3d
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from noise2same import network


class DonutMask(nn.Module):
    def __init__(self, n_dim: int = 2, in_channels: int = 1):
        """
        Local average excluding the center pixel
        :param n_dim:
        :param in_channels:
        """
        super(DonutMask, self).__init__()
        assert n_dim in (2, 3)
        self.n_dim = n_dim

        kernel = (
            np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
            if n_dim == 2
            else np.array(
                [
                    [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                    [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]],
                    [[0, 0.5, 0], [0.5, 1.0, 0.5], [0, 0.5, 0]],
                ]
            )
        )
        kernel = kernel / kernel.sum()
        kernel = torch.from_numpy(kernel)[None, None]
        kernel = kernel.expand(in_channels, in_channels, -1, -1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: T) -> T:
        conv = conv2d if self.n_dim == 2 else conv3d
        # todo understand stride
        return conv(x, self.kernel, padding=1, stride=1)


class Noise2Same(nn.Module):
    def __init__(
        self,
        n_dim: int = 2,
        in_channels: int = 1,
        base_channels: int = 96,
        lambda_inv: float = 2.0,
        mask_percentage: float = 0.5,
        masking: str = "gaussian",
        noise_mean: float = 0,
        noise_std: float = 0.2,
        **kwargs: Any,
    ):
        """

        :param n_dim:
        :param in_channels:
        :param base_channels:
        :param lambda_inv:
        :param mask_percentage:
        :param masking:
        :param noise_mean:
        :param noise_std:
        """
        super(Noise2Same, self).__init__()
        assert masking in ("gaussian", "donut")
        self.n_dim = n_dim
        self.in_channels = in_channels
        self.lambda_inv = lambda_inv
        self.mask_percentage = mask_percentage
        self.masking = masking
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        # TODO customize with segmentation_models
        self.net = network.UNet(
            in_channels=in_channels, n_dim=n_dim, base_channels=base_channels, **kwargs
        )
        self.head = network.RegressionHead(
            in_channels=base_channels, out_channels=in_channels
        )

        self.mask_kernel = DonutMask(n_dim=n_dim, in_channels=in_channels)

    def forward_full(self, x: T, mask: T) -> Tuple[T, T]:
        """
        Make two forward passes: with mask and without mask
        :param x:
        :param mask:
        :return: tuple of tensors: output for masked input, output for raw input
        """
        out_mask = self.forward_masked(x, mask)
        out_raw = self.forward(x)
        return out_mask, out_raw

    def forward_masked(self, x: T, mask: T) -> T:
        """
        Mask the image according to selected masking, then do the forward pass:
        substitute with gaussian noise or local average excluding center pixel (donut)
        :param x:
        :param mask:
        :return:
        """
        noise = (
            torch.randn(*x.shape, device=x.device, requires_grad=False) * self.noise_std
            + self.noise_mean
            # np.random.normal(self.noise_mean, self.noise_std, x.shape)
            if self.masking == "gaussian"
            else self.mask_kernel(x)
        )
        x = (1 - mask) * x + mask * noise
        return self.forward(x)

    def forward(self, x: T, *args: Any, **kwargs: Any) -> T:
        """
        Plain raw forward pass without masking
        :param x:
        :return:
        """
        x = self.net(x)
        x = self.head(x)
        return x

    def compute_losses_from_output(
        self, x: T, mask: T, out_mask: T, out_raw: T
    ) -> Tuple[T, Dict[str, float]]:
        rec_mse = torch.mean(torch.square(out_raw - x))
        inv_mse = torch.sum(torch.square(out_raw - out_mask) * mask) / torch.sum(mask)
        bsp_mse = torch.sum(torch.square(x - out_mask) * mask) / torch.sum(mask)
        loss = rec_mse + self.lambda_inv * torch.sqrt(inv_mse)
        loss_log = {
            "loss": loss.item(),
            "rec_mse": rec_mse.item(),
            "inv_mse": inv_mse.item(),
            "bsp_mse": bsp_mse.item(),
        }
        return loss, loss_log

    def compute_losses(self, x: T, mask: T) -> Tuple[T, Dict[str, float]]:
        out_mask, out_raw = self.forward_full(x, mask)
        return self.compute_losses_from_output(x, mask, out_mask, out_raw)


class Trainer(object):
    def __init__(
        self,
        model: Noise2Same,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        checkpoint_path: str = "checkpoints",
        monitor: str = "val_rec_mse",
        check: bool = False,
        wandb_log: bool = True,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.monitor = monitor
        self.check = check
        if check:
            wandb_log = False
        self.wandb_log = wandb_log

        self.model.to(device)
        self.checkpoint_path.mkdir(parents=True, exist_ok=False)

    def one_epoch(
        self, loader: DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.train()
        iterator = tqdm(loader, desc="train")
        total_loss = Counter()
        images = {}

        for i, batch in enumerate(iterator):
            x = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            out_mask, out_raw = self.model.forward_full(x, mask)
            loss, loss_log = self.model.compute_losses_from_output(
                x, mask, out_mask, out_raw
            )

            # todo gradient accumulation
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss_log
            iterator.set_postfix({k: v / (i + 1) for k, v in total_loss.items()})

            if self.check and i > 3:
                break

            if i == len(iterator) - 1 or (self.check and i == 3):
                images = {
                    "input": x,
                    "out_mask": out_mask,
                    "out_raw": out_raw,
                }
                images = {
                    k: (v.detach().cpu() * batch["std"] + batch["mean"])
                    .numpy()
                    .clip(0, 255)
                    for k, v in images.items()
                }
        total_loss = {k: v / len(loader) for k, v in total_loss.items()}
        if self.scheduler is not None:
            total_loss["lr"] = self.scheduler.get_last_lr()[0]
        return total_loss, images

    @torch.no_grad()
    def validate(
        self, loader: DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.eval()
        iterator = tqdm(loader, desc="valid")

        total_loss = 0
        images = {}
        for i, batch in enumerate(iterator):
            x = batch["image"].to(self.device)
            out_raw = self.model(x)
            rec_mse = torch.mean(torch.square(out_raw - x))
            total_loss += rec_mse.item()
            iterator.set_postfix({"val_rec_mse": total_loss / (i + 1)})

            if self.check and i > 3:
                break

            if i == len(iterator) - 1 or (self.check and i == 3):
                images = {
                    "val_input": x,
                    "val_out_raw": out_raw,
                }
                images = {
                    k: (v.detach().cpu() * batch["std"] + batch["mean"])
                    .numpy()
                    .clip(0, 255)
                    for k, v in images.items()
                }

        return {"val_rec_mse": total_loss / len(loader)}, images

    @torch.no_grad()
    def inference(self, loader: DataLoader) -> List[np.ndarray]:
        self.model.eval()

        outputs = []
        iterator = tqdm(loader, desc="inference", position=0, leave=True)
        for i, batch in enumerate(iterator):
            x = batch["image"].to(self.device)
            out_raw = self.model(x).cpu() * batch["std"] + batch["mean"]
            outputs.append(out_raw.numpy().clip(0, 255))
        return outputs

    def fit(
        self,
        n_epochs: int,
        loader_train: DataLoader,
        loader_valid: Optional[DataLoader] = None,
    ) -> List[Dict[str, float]]:

        iterator = trange(n_epochs)
        history = []
        best_loss = np.inf

        # if self.wandb_log:
        #     wandb.watch(self.model)

        try:
            for i in iterator:
                loss, images = self.one_epoch(loader_train)
                if loader_valid is not None:
                    loss_valid, images_valid = self.validate(loader_valid)
                    loss.update(loss_valid)
                    images.update(images_valid)

                # Log training
                if self.wandb_log:
                    images_wandb = {
                        k: [wandb.Image(im) for im in v] for k, v in images.items()
                    }
                    wandb.log({**images_wandb, **loss})

                # Show progress
                iterator.set_postfix(loss)
                history.append(loss)
                # Save best model
                if self.monitor not in loss:
                    print(
                        f"Nothing to monitor! {self.monitor} not in recorded losses {list(loss.keys())}"
                    )
                    continue

                if loss[self.monitor] < best_loss:
                    print(
                        f"Saved best model by {self.monitor}: {loss[self.monitor]:.4e} < {best_loss:.4e}"
                    )
                    self.save_model()
                    best_loss = loss[self.monitor]

                if self.check and i > 3:
                    break

        except KeyboardInterrupt:
            print("Interrupted")

        return history

    def save_model(self, name: str = "model"):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            f"{self.checkpoint_path}/{name}.pth",
        )

    def load_model(self, path: Optional[str] = None):
        if path is None:
            path = self.checkpoint_path
        checkpoint = torch.load(path)
        for attr, state_dict in checkpoint.items():
            try:
                getattr(self, attr).load_state_dict(state_dict)
            except AttributeError:
                print(
                    f"Attribute {attr} is present in the checkpoint but absent in the class, do not load"
                )
