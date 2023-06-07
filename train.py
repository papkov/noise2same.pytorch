import os
import traceback
from pathlib import Path
from random import randint
from time import sleep

import hydra
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler

import evaluate
import noise2same.trainer
from noise2same import model, util
from noise2same.dataset.getter import get_dataset, get_test_dataset_and_gt
from noise2same.optimizer.esadam import ESAdam
from utils import parametrize_backbone_and_head
from noise2same.scheduler import ExponentialDecayScheduler


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:

    # trying to fix: unable to open shared memory object </torch_197398_0> in read-write mode
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # Prevent from writing from the same log folder
    sleep(randint(1, 5))

    if "backbone_name" not in cfg.keys():
        print("Please specify a backbone with `+backbone=name`")
        return

    if "experiment" not in cfg.keys():
        print("Please specify an experiment with `+experiment=name`")
        return

    print(OmegaConf.to_yaml(cfg))
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"
    print(f"Run backbone {cfg.backbone_name} on experiment {cfg.experiment}, work in {os.getcwd()}")
    cwd = Path(get_original_cwd())

    util.fix_seed(cfg.seed)

    # flatten 2-level config
    d_cfg = {}
    for group, group_dict in dict(cfg).items():
        if isinstance(group_dict, DictConfig):
            for param, value in dict(group_dict).items():
                d_cfg[f"{group}.{param}"] = value
        else:
            d_cfg[group] = group_dict

    if not cfg.check:
        wandb.init(project=cfg.project, config=d_cfg, settings=wandb.Settings(start_method="fork"))
        wandb.run.summary.update({'training_dir': os.getcwd()})

    # Data
    dataset_train, dataset_valid = get_dataset(cfg, cwd)
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
        n_samples_val = int(cfg.training.val_partition * len(dataset_valid))
        loader_valid = DataLoader(
            torch.utils.data.random_split(
                dataset_valid,
                [n_samples_val, len(dataset_valid) - n_samples_val],
                generator=torch.Generator().manual_seed(42)
            )[0],
            batch_size=cfg.training.val_batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    # Read PSF from dataset if available or by path
    psf = getattr(dataset_train, "psf", None)
    if psf is None and getattr(cfg, "psf", None) is not None:
        psf = cwd / cfg.psf.path
        print(f"Read PSF from {psf}")

    backbone, head = parametrize_backbone_and_head(cfg)

    # Model
    mdl = model.Noise2Same(
        n_dim=cfg.dataset.n_dim,
        in_channels=cfg.dataset.n_channels,
        psf=psf,
        psf_size=cfg.psf.psf_size if "psf" in cfg else None,
        psf_pad_mode=cfg.psf.psf_pad_mode if "psf" in cfg else None,
        psf_fft=cfg.psf.psf_fft if "psf" in cfg else None,
        backbone=backbone,
        head=head,
        **cfg.model,
    )

    if torch.cuda.device_count() > 1:
        print(f'Using data parallel with {torch.cuda.device_count()} GPUs')
        mdl = torch.nn.DataParallel(mdl)

    # Optimization
    if cfg.optim.optimizer == "adam":
        optimizer = torch.optim.Adam(mdl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == "esadam":
        optimizer = ESAdam(mdl.parameters(), lr=cfg.optim.lr)
    else:
        raise ValueError(f"Unknown optimizer {cfg.optim.optimizer}")

    if cfg.optim.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            cfg.training.steps,
            eta_min=cfg.optim.eta_min,
        )
    else:
        scheduler = ExponentialDecayScheduler(
            optimizer,
            decay_rate=cfg.optim.decay_rate,
            decay_steps=cfg.optim.decay_steps,
            staircase=cfg.optim.staircase,
        )

    # Trainer
    trainer = noise2same.trainer.Trainer(
        model=mdl,
        optimizer=optimizer,
        scheduler=scheduler,
        check=cfg.check,
        monitor=cfg.training.monitor,
        experiment=cfg.experiment,
        amp=cfg.training.amp,
    )

    n_epochs = cfg.training.steps // cfg.training.steps_per_epoch
    try:
        history = trainer.fit(
            n_epochs, loader_train, loader_valid if cfg.training.validate else None
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    except RuntimeError:
        if not cfg.check:
            wandb.run.summary["error"] = "RuntimeError"
        traceback.print_exc()

    if cfg.evaluate:
        test_dataset, ground_truth = get_test_dataset_and_gt(cfg, cwd)

        scores = evaluate.evaluate(trainer.evaluator, ground_truth, cfg.experiment, cwd, Path(os.getcwd()),
                                   dataset=test_dataset, half=cfg.training.amp, num_workers=cfg.training.num_workers)

        if not cfg.check:
            wandb.log(scores)
            wandb.run.summary.update(scores)

    if not cfg.check:
        wandb.finish()


if __name__ == "__main__":
    main()
