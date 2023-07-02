import os
import traceback

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler

import evaluate
import noise2same.trainer
from noise2same import util
from noise2same.dataset.getter import expand_dataset_cfg
from noise2same.psf.psf_convolution import instantiate_psf
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig) -> None:
    # trying to fix: unable to open shared memory object </torch_197398_0> in read-write mode
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # Expand dataset config for validation and test datasets to extend train dataset
    expand_dataset_cfg(cfg)

    # Check if all necessary arguments are specified
    for arg in ["backbone", "experiment", "denoiser"]:
        if arg not in cfg.keys():
            log.info(f"Please specify a {arg} with `+{arg}=name`")
            return

    log.info(OmegaConf.to_yaml(cfg))
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"
    log.info(f"Run backbone {cfg.backbone_name} on experiment {cfg.experiment}, work in {os.getcwd()}")

    # Make training deterministic
    util.fix_seed(cfg.seed)

    # Start WandB logging
    if not cfg.check:
        wandb.init(project=cfg.project, config=util.flatten_config(cfg), settings=wandb.Settings(start_method="fork"))
        wandb.run.summary.update({'training_dir': os.getcwd()})
    else:
        os.environ["HYDRA_FULL_ERROR"] = "1"

    # Data
    dataset_train = instantiate(cfg.dataset_train)
    dataset_valid = instantiate(cfg.dataset_valid) if 'dataset_valid' in cfg else None

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

    # Model
    backbone = instantiate(cfg.backbone)
    head = instantiate(cfg.head)
    denoiser = instantiate(cfg.denoiser, backbone=backbone, head=head, **instantiate_psf(cfg, dataset_train))

    if torch.cuda.device_count() > 1:
        log.info(f'Using data parallel with {torch.cuda.device_count()} GPUs')
        denoiser = torch.nn.DataParallel(denoiser)

    # Optimization
    # TODO test instantiation for common configs
    optimizer = instantiate(cfg.optimizer, denoiser.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)

    # Trainer
    trainer = noise2same.trainer.Trainer(
        denoiser=denoiser,
        optimizer=optimizer,
        scheduler=scheduler,
        check=cfg.check,
        monitor=cfg.training.monitor,
        experiment=cfg.experiment,
        amp=cfg.training.amp,
    )

    n_epochs = cfg.training.steps // cfg.training.steps_per_epoch
    try:
        _ = trainer.fit(n_epochs, loader_train, loader_valid)
    except KeyboardInterrupt:
        log.info("Training interrupted")
    except RuntimeError:
        if not cfg.check:
            wandb.run.summary["error"] = "RuntimeError"
        traceback.print_exc()

    if 'evaluate' in cfg:
        dataset_test = instantiate(cfg.dataset_test)
        factory = instantiate(cfg.factory_test) if 'factory_test' in cfg else None

        scores = evaluate.evaluate(
            evaluator=trainer.evaluator,
            dataset=dataset_test,
            cfg=cfg,
            factory=factory,
            train_dir=os.getcwd(),
        )

        if not cfg.check:
            wandb.log(scores)
            wandb.run.summary.update(scores)

    if not cfg.check:
        wandb.finish()


if __name__ == "__main__":
    util.register_config_resolvers()
    main()
