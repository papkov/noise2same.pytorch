import glob
import os
import datetime
from pathlib import Path
from pprint import pprint

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm.auto import tqdm

from noise2same import model, util
from noise2same.dataset.getter import (
    get_planaria_dataset_and_gt,
    get_test_dataset_and_gt,
)
from noise2same.evaluator import Evaluator
from noise2same.backbone.utils import parametrize_backbone_and_head


def get_ground_truth_and_predictions(
    evaluator: Evaluator,
    experiment: str,
    ground_truth: np.ndarray,
    cwd: Path,
    loader: DataLoader = None,
    dataset: Dataset = None,
    half: bool = False
):
    if experiment in ("bsd68", "fmd", "hanzi", "sidd", "synthetic", "synthetic_grayscale"):
        add_blur_and_noise = getattr(dataset, "add_blur_and_noise", False)
        if add_blur_and_noise:
            print("Validate for deconvolution")
        predictions, _ = evaluator.inference(loader, half=half)
    elif experiment in ("imagenet",):
        predictions, indices = evaluator.inference(loader, half=half, empty_cache=True)
        ground_truth = [ground_truth[i] for i in indices]
    elif experiment in ("microtubules",):
        predictions, _ = evaluator.inference_single_image_dataset(
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
    predictions: np.ndarray,
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
        scores = util.calculate_scores(ground_truth, predictions, normalize_pairs=True)
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
    else:
        raise ValueError
    return scores


def evaluate(
    evaluator: Evaluator,
    dataset: Dataset,
    ground_truth: np.ndarray,
    experiment: str,
    cwd: Path,
    train_dir: Path,
    num_workers: int = None,
    half: bool = False,
    save_results: bool = True,
    verbose: bool = True,
):
    loader = DataLoader(
        dataset,
        batch_size=1,  # todo customize
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    ground_truth, predictions = get_ground_truth_and_predictions(
        evaluator, experiment, ground_truth, cwd, loader, dataset, half
    )

    scores = get_scores(ground_truth, predictions, experiment)
    scores = pd.DataFrame(scores)

    if experiment in ("synthetic", "synthetic_grayscale",):
        # Label each score with its dataset name and repeat id
        # by default {"kodak": 10, "bsd300": 3, "set14": 20} but the code below generalizes to any number of repeats
        dataset_name = []
        repeat_id = []
        repeat = 0
        assert isinstance(dataset, ConcatDataset)
        for ds in dataset.datasets:
            assert isinstance(ds, ConcatDataset)
            for repeat_ds in ds.datasets:
                repeat_id += [repeat] * len(repeat_ds)
                dataset_name += [repeat_ds.name] * len(repeat_ds)
                repeat += 1
            repeat = 0
        scores = scores.assign(dataset_name=dataset_name, repeat_id=repeat_id)
    evaluation_dir = train_dir / f'evaluate' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if save_results:
        print("Saving results to", evaluation_dir)
        scores.to_csv(evaluation_dir / "scores.csv")
        np.savez(evaluation_dir / "predictions.npz", **predictions)

    if experiment in ("planaria",):
        scores = scores.groupby("c").mean()
    elif experiment in ("synthetic", "synthetic_grayscale",):
        if verbose:
            print("\nBefore averaging over repeats:")
            pprint(scores.groupby(["dataset_name", "repeat_id"]).mean())
        scores = scores.groupby("dataset_name").mean().drop(columns="repeat_id")
    else:
        scores = scores.mean()

    if verbose:
        print("\nEvaluation results:")
        pprint(scores)

    scores = scores.to_dict()
    if experiment in ("synthetic", "synthetic_grayscale", "planaria", ):
        # Flatten scores dict as "metric.dataset" to make it compatible with wandb
        scores = {f"{metric}.{dataset_name}": score for metric, dataset_dict in scores.items()
                  for dataset_name, score in dataset_dict.items()}
    return scores


def main(train_dir: Path, checkpoint: str = 'last', other_args: list = None) -> None:

    cfg = OmegaConf.load(f'{train_dir}/.hydra/config.yaml')
    if other_args is not None:
        cfg.merge_with_dotlist(other_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"

    print(f"Evaluate backbone {cfg.backbone_name} on experiment {cfg.experiment}, work in {train_dir}")

    cwd = Path(os.getcwd())

    dataset, ground_truth = None, None
    if cfg.experiment not in ("planaria",):
        # For some datasets we need custom loading
        dataset, ground_truth = get_test_dataset_and_gt(cfg)

    backbone, head = parametrize_backbone_and_head(cfg)

    mdl = model.Noise2Same(
        n_dim=cfg.dataset.n_dim,
        in_channels=cfg.dataset.n_channels,
        psf=cfg.psf.path if "psf" in cfg else None,
        psf_size=cfg.psf.psf_size if "psf" in cfg else None,
        psf_pad_mode=cfg.psf.psf_pad_mode if "psf" in cfg else None,
        backbone=backbone,
        head=head,
        **cfg.model,
    )

    checkpoint_path = train_dir / Path(f"checkpoints/model{'_last' if checkpoint == 'last' else ''}.pth")

    # Run evaluation
    half = getattr(cfg, "amp", False)
    masked = getattr(cfg, "masked", False)
    evaluator = Evaluator(mdl, checkpoint_path=checkpoint_path, masked=masked)
    evaluate(
        evaluator, dataset, ground_truth, cfg.experiment, cwd, train_dir, half=half,
        num_workers=cfg.training.num_workers
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir", required=True,
                        help="Path to hydra train directory")
    parser.add_argument("--checkpoint", choices=["last", "best"],
                        default="last", help="The checkpoint to evaluate, 'last' or 'best'")
    args, unknown_args = parser.parse_known_args()
    main(Path(args.train_dir), args.checkpoint, unknown_args)
