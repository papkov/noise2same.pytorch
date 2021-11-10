from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from noise2same.evaluator import Evaluator
from noise2same.model import Noise2Same
from noise2same.util import load_checkpoint_to_module


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
        self.evaluator = Evaluator(model=model, device=device)

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
                    "out_mask": out_mask["image"],
                    "out_raw": out_raw["image"],
                }
                images = {
                    k: np.moveaxis(
                        (v.detach().cpu() * batch["std"] + batch["mean"]).numpy(), 1, -1
                    )
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
            out_raw = self.model(x)["image"]
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
                    k: np.moveaxis(
                        (v.detach().cpu() * batch["std"] + batch["mean"]).numpy(), 1, -1
                    )
                    for k, v in images.items()
                }

        return {"val_rec_mse": total_loss / len(loader)}, images

    def inference(self, **kwargs: Any):
        return self.evaluator.inference(**kwargs)

    def inference_single_image_dataset(self, **kwargs: Any):
        return self.evaluator.inference_single_image_dataset(**kwargs)

    def inference_single_image_tensor(self, **kwargs: Any):
        return self.evaluator.inference_single_image_tensor(**kwargs)

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
                        # limit the number of uploaded images
                        # if image is 3d, reduce it
                        k: [
                            wandb.Image(im.max(0) if self.model.n_dim == 3 else im)
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
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            f"{self.checkpoint_path}/{name}.pth",
        )

    def load_model(self, path: Optional[str] = None):
        if path is None:
            path = self.checkpoint_path
        load_checkpoint_to_module(self, path)
