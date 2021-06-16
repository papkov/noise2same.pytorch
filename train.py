import os
from pathlib import Path

import hydra
import torch
import wandb
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from noise2same import dataset, models, util


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


@hydra.main(config_path="config", config_name="default.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"
    util.fix_seed(cfg.seed)
    cwd = Path(get_original_cwd())

    if not cfg.check:
        wandb.init(project=cfg.project, config=dict(cfg))

    # todo parametrize everything in cfg
    if cfg.data.name.lower() == "bsd68":
        dataset_train = dataset.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/",
            mode="train",
            transforms=dataset.training_augmentations(crop=cfg.data.crop),
        )
        dataset_valid = dataset.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/", mode="val"
        )
    else:
        # todo add other datasets
        raise ValueError

    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=4,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # todo imply n_dim and in_channels from data
    model = models.Noise2Same(n_dim=2, in_channels=1, **cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=exponential_decay(
            decay_rate=cfg.optim.decay_rate,
            decay_steps=cfg.optim.decay_steps,
            staircase=cfg.optim.staircase,
        ),
    )

    trainer = models.Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        check=cfg.check,
    )

    history = trainer.fit(cfg.train.n_epochs, loader_train, loader_valid)

    if not cfg.check:
        wandb.finish()


if __name__ == "__main__":
    main()
