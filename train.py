import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from noise2same import dataset, models, util


@hydra.main(config_path="config", config_name="default.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"
    util.fix_seed(cfg.seed)
    cwd = Path(hydra.utils.get_original_cwd())

    if not cfg.check:
        wandb.init(project=cfg.project, config=dict(cfg))

    # todo parametrize everything in cfg
    if cfg.data.lower() == "bsd68":
        dataset_train = dataset.BSD68DatasetPrepared(
            path=cwd / "data/BSD68/", mode="train"
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
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    n_iter = cfg.train.n_epochs * len(loader_train)  # number of loader iterations
    # todo imply n_dim and in_channels from data
    model = models.Noise2Same(n_dim=2, in_channels=1, **cfg.model)
    # todo parametrize as hydra init
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)

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
