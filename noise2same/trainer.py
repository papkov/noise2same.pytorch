from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from noise2same.evaluator import Evaluator
from noise2same.model import Noise2Same
from noise2same.util import (
    detach_to_np,
    load_checkpoint_to_module,
    normalize_zero_one_dict,
)
from evaluate import get_scores


class Trainer(object):
    def __init__(
        self,
        model: Noise2Same,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        checkpoint_path: str = "checkpoints",
        monitor: str = "val_rec_mse",
        experiment: str = None,
        check: bool = False,
        wandb_log: bool = True,
        amp: bool = False,
        info_padding: bool = False,
    ):

        self.model = model
        self.inner_model = model if not isinstance(model, torch.nn.DataParallel) else model.module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.monitor = monitor
        self.experiment = experiment
        self.check = check
        if check:
            wandb_log = False
        self.wandb_log = wandb_log
        self.info_padding = info_padding

        self.model.to(device)
        self.checkpoint_path.mkdir(parents=True, exist_ok=False)
        self.evaluator = Evaluator(model=self.inner_model, device=device)

        self.amp = amp
        self.scaler = GradScaler() if amp else None

    def optimizer_scheduler_step(self, loss: torch.Tensor):
        """
        Step the optimizer and scheduler given the loss
        :param loss:
        :return:
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def one_epoch(
        self, loader: DataLoader
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.train()
        iterator = tqdm(loader, desc="train")
        total_loss = Counter()
        images = {}
        for i, batch in enumerate(iterator):
            x = batch["image"].to(self.device)
            ground_truth = None if "ground_truth" not in batch else batch["ground_truth"].to(self.device)
            mask = batch["mask"].to(self.device)
            self.optimizer.zero_grad()

            # todo gradient accumulation
            with autocast(enabled=self.amp):
                if self.info_padding:
                    # provide full size image to avoid zero padding
                    padding = [
                        (b, a)
                        for b, a in zip(
                            loader.dataset.tiler.margin_start,
                            loader.dataset.tiler.margin_end,
                        )
                    ] + [(0, 0)]
                    full_size_image = np.pad(loader.dataset.image, padding)
                    full_size_image = torch.from_numpy(
                        np.moveaxis(full_size_image, -1, 0)
                    ).to(self.device)
                    out_mask, out_raw = self.model.forward(
                        x, mask=mask, crops=batch["crop"], full_size_image=full_size_image
                    )
                else:
                    try:
                        out_mask, out_raw = self.model.forward(x, mask=mask)
                    except RuntimeError as e:
                        raise RuntimeError(f"Batch {x.shape} failed on device {x.device}") from e

                loss, loss_log = self.inner_model.compute_losses_from_output(
                    x, mask, out_mask, out_raw, ground_truth
                )

                reg_loss, reg_loss_log = self.inner_model.compute_regularization_loss(
                    out_mask if out_raw is None else out_raw,
                    mean=batch["mean"].to(self.device),
                    std=batch["std"].to(self.device),
                    max_value=x.max(),
                )

                if reg_loss_log:
                    loss += reg_loss
                    loss_log.update(reg_loss_log)
            self.optimizer_scheduler_step(loss)
            total_loss += loss_log
            iterator.set_postfix({k: v / (i + 1) for k, v in total_loss.items()})

            if self.check and i > 3:
                break

            # Log last batch of images
            if i == len(iterator) - 1 or (self.check and i == 3):
                images = {
                    "input": x
                }

                if out_mask is not None:
                    images["out_mask"] = out_mask["image"]

                if out_raw is not None:
                    images["out_raw"] = out_raw["image"]

                if out_mask is not None and "deconv" in out_mask:
                    images["out_mask_deconv"] = out_mask["deconv"]
                if out_raw is not None and "deconv" in out_raw:
                    images["out_raw_deconv"] = out_raw["deconv"]

                images = detach_to_np(images, mean=batch["mean"], std=batch["std"])
                images = normalize_zero_one_dict(images)

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
        val_mse_log = []
        images = {}
        for i, batch in enumerate(iterator):
            x = batch["image"].to(self.device)

            with autocast(enabled=self.amp):
                out_raw = self.model(x)[1]["image"]

            if "ground_truth" in batch.keys():
                val_mse = torch.mean(torch.square(out_raw - batch["ground_truth"].to(self.device)))
                val_mse_log.append(val_mse.item())

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
                images = detach_to_np(images, mean=batch["mean"], std=batch["std"])
                images = normalize_zero_one_dict(images)
        if len(val_mse_log) > 0:
            return {"val_rec_mse": total_loss / len(loader), "val_mse": np.mean(val_mse_log)}, images
        return {"val_rec_mse": total_loss / len(loader)}, images

    def inference(self, *args: Any, **kwargs: Any):
        return self.evaluator.inference(*args, **kwargs)

    def inference_single_image_dataset(self, *args: Any, **kwargs: Any):
        return self.evaluator.inference_single_image_dataset(*args, **kwargs)

    def inference_single_image_tensor(self, *args: Any, **kwargs: Any):
        return self.evaluator.inference_single_image_tensor(*args, **kwargs)

    def fit(
        self,
        n_epochs: int,
        loader_train: DataLoader,
        loader_valid: Optional[DataLoader] = None,
    ) -> List[Dict[str, float]]:

        iterator = trange(n_epochs, position=0, leave=True)
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
                        # limit the number of uploaded images
                        # if image is 3d, reduce it
                        k: [
                            wandb.Image(im.max(0) if self.inner_model.n_dim == 3 else im)
                            for im in v[:4]
                        ]
                        for k, v in images.items()
                    }
                    wandb.log({**images_wandb, **loss})

                # Show progress
                iterator.set_postfix(loss)
                history.append(loss)

                # Save last model
                self.save_model("model_last")

                if self.check and i > 3:
                    break

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

        except KeyboardInterrupt:
            print("Interrupted")

        return history

    def save_model(self, name: str = "model"):
        torch.save(
            {
                "model": self.inner_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            f"{self.checkpoint_path}/{name}.pth",
        )

    def load_model(self, path: Optional[str] = None):
        if path is None:
            path = self.checkpoint_path / "model.pth"
        load_checkpoint_to_module(self, path)
