import os
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler, Subset

import noise2same.trainer
from noise2same import model, util
from noise2same.dataset import bsd68, hanzi, imagenet, planaria
from noise2same.dataset.util import training_augmentations_2d, training_augmentations_3d


def exponential_decay(
    decay_rate: float = 0.5, decay_steps: int = 5e3, staircase: bool = True
):
    """
    Lambda for torch.optim.lr_scheduler.LambdaLR mimicking tf.train.exponential_decay:
    decayed_learning_rate = learning_rate *
                            decay_rate ^ (global_step / decay_steps)

    :param decay_rate: float, multiplication factor
    :param decay_steps: int, how many steps to make to multiply by decay_rate
    :param staircase: bool, integer division global_step / decay_steps
    :return: lambda(epoch)
    """

    def _lambda(epoch: int):
        exp = epoch / decay_steps
        if staircase:
            exp = int(exp)
        return decay_rate ** exp

    return _lambda


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    if "name" not in cfg.keys():
        print("Please specify an experiment with `+experiment=name`")
        return

    print(OmegaConf.to_yaml(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"

    util.fix_seed(cfg.seed)
    cwd = Path(get_original_cwd())

    if not cfg.check:
        wandb.init(project=cfg.project, config=dict(cfg))

    if cfg.name.lower() == "bsd68":
        dataset_train = bsd68.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/",
            mode="train",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
        )
        dataset_valid = None
        if cfg.training.validate:
            dataset_valid = bsd68.BSD68DatasetPrepared(
                path=cwd / "data/BSD68/", mode="val"
            )
    elif cfg.name.lower() == "hanzi":
        dataset_train = hanzi.HanziDatasetPrepared(
            path=cwd / "data/Hanzi/tiles",
            mode="training",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            version=cfg.data.version,
            noise_level=cfg.data.noise_level,
        )
        dataset_valid = None
        if cfg.training.validate:
            dataset_valid = hanzi.HanziDatasetPrepared(
                path=cwd / "data/Hanzi/tiles",
                mode="validation",
                version=cfg.data.version,
                noise_level=cfg.data.noise_level,
            )
    elif cfg.name.lower() == "imagenet":
        dataset_train = imagenet.ImagenetDatasetPrepared(
            path=cwd / "data/ImageNet",
            mode="train",
            transforms=training_augmentations_2d(crop=cfg.training.crop),
            version=cfg.data.version,
        )
        dataset_valid = None
        if cfg.training.validate:
            dataset_valid = imagenet.ImagenetDatasetPrepared(
                path=cwd / "data/ImageNet",
                mode="val",
                version=cfg.data.version,
            )
    elif cfg.name.lower() == "planaria":
        dataset_train = planaria.PlanariaDatasetPrepared(
            path=cwd / "data/Denoising_Planaria",
            mode="train",
            transforms=training_augmentations_3d(),
        )
        dataset_valid = None
        if cfg.training.validate:
            dataset_valid = planaria.PlanariaDatasetPrepared(
                path=cwd / "data/Denoising_Planaria",
                mode="val",
            )
    else:
        # todo add other datasets
        raise ValueError

    num_samples = cfg.training.batch_size * cfg.training.steps_per_epoch
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler=RandomSampler(dataset_train, replacement=True, num_samples=num_samples),
        pin_memory=True,
        drop_last=True,
    )

    loader_valid = None
    if cfg.training.validate:
        loader_valid = DataLoader(
            dataset_valid,
            batch_size=4,
            num_workers=cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    mdl = model.Noise2Same(
        n_dim=cfg.data.n_dim,
        in_channels=cfg.data.n_channels,
        **cfg.model,
    )
    optimizer = torch.optim.Adam(mdl.parameters(), lr=cfg.optim.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=exponential_decay(
            decay_rate=cfg.optim.decay_rate,
            decay_steps=cfg.optim.decay_steps,
            staircase=cfg.optim.staircase,
        ),
    )

    trainer = noise2same.trainer.Trainer(
        model=mdl,
        optimizer=optimizer,
        scheduler=scheduler,
        check=cfg.check,
        monitor=cfg.training.monitor,
    )

    n_epochs = cfg.training.steps // cfg.training.steps_per_epoch
    history = trainer.fit(
        n_epochs, loader_train, loader_valid if cfg.training.validate else None
    )

    if not cfg.check:
        wandb.finish()


if __name__ == "__main__":
    main()
