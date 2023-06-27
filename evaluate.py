import datetime
import glob
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm.auto import tqdm

from noise2same import util
from noise2same.dataset.abc import AbstractNoiseDataset
from noise2same.dataset.getter import expand_dataset_cfg
from noise2same.dataset.getter import (
    get_planaria_dataset_and_gt,
)
from noise2same.dataset.tiling import TiledImageFactory
from noise2same.evaluator import Evaluator
from noise2same.psf.psf_convolution import instantiate_psf

log = logging.getLogger(__name__)


def get_ground_truth_and_predictions(
        evaluator: Evaluator,
        experiment: str,
        ground_truth: np.ndarray,
        cwd: Path,
        loader: DataLoader = None,
        dataset: Dataset = None,
        half: bool = False
):
    if experiment in ("bsd68", "fmd", "hanzi", "sidd", "synthetic", "synthetic_grayscale", "hela_shallow"):
        add_blur_and_noise = getattr(dataset, "add_blur_and_noise", False)
        if add_blur_and_noise:
            log.info("Validate for deconvolution")
        predictions, _ = evaluator.inference(loader, half=half)
    elif experiment in ("imagenet",):
        predictions, indices = evaluator.inference(loader, half=half, empty_cache=True)
        ground_truth = [ground_truth[i] for i in indices]
    elif experiment in ("microtubules",):
        predictions = evaluator.inference_single_image_dataset(
            dataset, half=half, batch_size=1
        )
    elif experiment in ("planaria",):
        files = sorted(
            glob.glob(str(cwd / "data/Denoising_Planaria/test_data/GT/*.tif"))
        )
        predictions = {"c1": [], "c2": [], "c3": [], "y": []}
        for f in tqdm(files):
            datasets, gt = get_planaria_dataset_and_gt(f)
            predictions["y"].append(gt)
            for c in range(1, 4):
                predictions[f"c{c}"].append(
                    evaluator.inference_single_image_dataset(
                        datasets[f"c{c}"], half=half, batch_size=1
                    )
                )
    else:
        raise ValueError

    # Rearrange predictions List[Dict[str, array]] -> Dict[str, List[array]]
    if experiment not in ("planaria", "microtubules"):
        predictions = {k: [d[k].squeeze() for d in predictions] for k in predictions[0]}

    return ground_truth, predictions


def get_scores(
        ground_truth: np.ndarray,
        predictions: Dict[str, np.ndarray],
        experiment: str
):
    # Calculate scores
    if experiment in ("bsd68",):
        scores = [
            util.calculate_scores(gtx, pred, data_range=255)
            for gtx, pred in zip(ground_truth, predictions["image"])
        ]
    elif experiment in ("synthetic", "synthetic_grayscale", "sidd", "fmd",):
        scale = 255 if experiment.startswith("synthetic") else 1
        multichannel = experiment in ("synthetic", "sidd")
        scores = [
            # https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/2fff2978/train.py#L446
            # SSIM is not exactly the same as the original Neighbor2Neighbor implementation,
            # because skimage uses padding (which is more fair), while the original implementation crops the borders.
            # However, the difference is negligible (<0.001 in their favor).
            util.calculate_scores(gtx.astype(np.float32),
                                  np.clip(pred * scale + 0.5, 0, 255).astype(np.uint8).astype(np.float32),
                                  data_range=255,
                                  multichannel=multichannel,
                                  gaussian_weights=True,
                                  )
            for gtx, pred in zip(ground_truth, predictions["image"])
        ]
    elif experiment in ("hanzi",):
        scores = [
            util.calculate_scores(gtx * 255, pred, data_range=255, scale=True)
            for gtx, pred in zip(ground_truth, predictions["image"])
        ]
    elif experiment in ("imagenet",):
        scores = [
            util.calculate_scores(
                gtx,
                pred,
                data_range=255,
                scale=True,
                multichannel=True,
            )
            for gtx, pred in zip(ground_truth, predictions["image"])
        ]
    elif experiment in ("microtubules",):
        scores = [util.calculate_scores(ground_truth, predictions["image"], normalize_pairs=True)]
    elif experiment in ("planaria",):
        scores = []
        for c in range(1, 4):
            scores_c = [
                util.calculate_scores(gt, x, normalize_pairs=True)
                for gt, x in tqdm(
                    zip(predictions["y"], predictions[f"c{c}"]),
                    total=len(predictions["y"]),
                )
            ]
            scores.append(pd.DataFrame(scores_c).assign(c=c))
        scores = pd.concat(scores)
    elif experiment in ("hela_shallow",):
        scores = [
            util.calculate_scores(gtx, pred, normalize_pairs=True)
            for gtx, pred in zip(ground_truth, predictions["image"])
        ]
    else:
        raise ValueError
    return scores


def evaluate(
        evaluator: Evaluator,
        dataset: AbstractNoiseDataset,
        cfg: DictConfig,
        factory: Optional[TiledImageFactory] = None,
        train_dir: Optional[str] = None,
        save_results: bool = True,
        verbose: bool = True,
        keep_images: bool = False,
        metrics: Tuple[str, ...] = ("rmse", "psnr", "ssim"),
):
    train_dir = train_dir or ''
    log.info(f"Evaluate dataset {str(dataset)} for key {cfg.evaluate.key} in train_dir {train_dir}")
    scores = evaluator.evaluate(dataset, factory,
                                num_workers=cfg.training.num_workers,
                                half=cfg.training.amp,
                                empty_cache=False,
                                key=cfg.evaluate.key,
                                keep_images=keep_images,
                                metrics=metrics,
                                )
    # TODO do not create a list of None if there are no images
    predictions = {cfg.evaluate.key: [s.pop(cfg.evaluate.key, None) for s in scores]}
    scores = pd.DataFrame(scores)

    # Label each score with its dataset name and repeat id
    datasets = [dataset] if not isinstance(dataset, ConcatDataset) else dataset.datasets
    scores = scores.assign(
        dataset_name=np.concatenate([[str(ds)] * len(ds) for ds in datasets]),
        repeat_id=np.concatenate([np.arange(len(ds)) // (len(ds) // ds.n_repeats) for ds in datasets])
    )
    evaluation_dir = Path(train_dir) / f'evaluate' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if save_results:
        log.info(f"Saving results to {evaluation_dir / 'scores.csv'}")
        scores.to_csv(evaluation_dir / "scores.csv")
        if keep_images:
            np.savez(evaluation_dir / "predictions.npz", **predictions)

    if verbose and any(ds.n_repeats > 1 for ds in datasets):
        log.info("\nBefore averaging over repeats:\n" +
                 pformat(scores.groupby(["dataset_name", "repeat_id"]).mean()))
    scores = scores.groupby("dataset_name").mean().drop(columns="repeat_id")

    if verbose:
        log.info("\nEvaluation results:\n" + pformat(scores))

    scores = scores.to_dict()
    # Flatten scores dict as "metric.dataset" to make it compatible with wandb
    scores = {f"{metric}{'.'+dataset_name if len(datasets) > 1 else ''}": score
              for metric, dataset_dict in scores.items()
              for dataset_name, score in dataset_dict.items()}
    return scores


def main(train_dir: Path, checkpoint: str = 'last', other_args: list = None) -> None:
    cfg = OmegaConf.load(f'{train_dir}/.hydra/config.yaml')
    if other_args is not None:
        cfg.merge_with_dotlist(other_args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"

    log.info(f"Evaluate backbone {cfg.backbone_name} on experiment {cfg.experiment}, work in {train_dir}")

    cfg.cwd = Path(os.getcwd())
    OmegaConf.resolve(cfg)
    # Expand dataset config for validation and test datasets to extend train dataset
    expand_dataset_cfg(cfg)
    log.info(OmegaConf.to_yaml(cfg))

    backbone = instantiate(cfg.backbone)
    head = instantiate(cfg.head)
    factory = instantiate(cfg.factory_test) if 'factory_test' in cfg else None
    dataset = instantiate(cfg.dataset_test)
    denoiser = instantiate(cfg.denoiser, backbone=backbone, head=head, **instantiate_psf(cfg, dataset))

    checkpoint_path = train_dir / Path(f"checkpoints/model{'_last' if checkpoint == 'last' else ''}.pth")

    # Run evaluation
    evaluator = Evaluator(denoiser, checkpoint_path=checkpoint_path)
    _ = evaluate(
        evaluator=evaluator,
        dataset=dataset,
        cfg=cfg,
        factory=factory,
        train_dir=train_dir,
        verbose=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir", required=True,
                        help="Path to hydra train directory")
    parser.add_argument("--checkpoint", choices=["last", "best"],
                        default="last", help="The checkpoint to evaluate, 'last' or 'best'")
    args, unknown_args = parser.parse_known_args()

    logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO,
                        filename=f'{args.train_dir}/evaluate.log')
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    log.info('Start evaluation')

    util.register_config_resolvers()
    main(Path(args.train_dir), args.checkpoint, unknown_args)
