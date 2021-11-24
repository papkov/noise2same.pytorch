import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler, Subset

import noise2same.trainer
from noise2same import model, util
from noise2same.dataset.getter import get_dataset
from noise2same.optimizers.esadam import ESAdam


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
    print(f"Run experiment {cfg.name}, work in {os.getcwd()}")

    util.fix_seed(cfg.seed)

    if not cfg.check:
        wandb.init(project=cfg.project, config=dict(cfg))

    # Data
    dataset_train, dataset_valid = get_dataset(cfg)
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

    # Model
    # TODO less ugly way to parametrize psf
    mdl = model.Noise2Same(
        n_dim=cfg.data.n_dim,
        in_channels=cfg.data.n_channels,
        psf=cfg.psf.path if "psf" in cfg else None,
        psf_size=cfg.psf.psf_size if "psf" in cfg else None,
        psf_pad_mode=cfg.psf.psf_pad_mode if "psf" in cfg else None,
        skip_method=cfg.network.skip_method,
        **cfg.model,
    )

    # Optimization
    if cfg.optim.optimizer == "adam":
        optimizer = torch.optim.Adam(mdl.parameters(), lr=cfg.optim.lr)
    elif cfg.optim.optimizer == "esadam":
        optimizer = ESAdam(mdl.parameters(), lr=cfg.optim.lr)
    else:
        raise ValueError(f"Unknown optimizer {cfg.optim.optimizer}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=exponential_decay(
            decay_rate=cfg.optim.decay_rate,
            decay_steps=cfg.optim.decay_steps,
            staircase=cfg.optim.staircase,
        ),
    )

    # Trainer
    trainer = noise2same.trainer.Trainer(
        model=mdl,
        optimizer=optimizer,
        scheduler=scheduler,
        check=cfg.check,
        monitor=cfg.training.monitor,
        amp=cfg.training.amp,
    )

    n_epochs = cfg.training.steps // cfg.training.steps_per_epoch
    history = trainer.fit(
        n_epochs, loader_train, loader_valid if cfg.training.validate else None
    )

    if not cfg.check:
        wandb.finish()


if __name__ == "__main__":
    main()
