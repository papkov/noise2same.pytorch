import glob
import os
from pathlib import Path
from pprint import pprint

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from noise2same import model, util
from noise2same.dataset.getter import (
    get_planaria_dataset_and_gt,
    get_test_dataset_and_gt,
)
from noise2same.evaluator import Evaluator
from utils import parametrize_backbone_and_head


def evaluate(
    evaluator: Evaluator,
    dataset: Dataset,
    ground_truth: np.ndarray,
    experiment: str,
    num_workers: int,
    cwd: Path,
    half: bool = False,
):
    loader = None
    if experiment.lower() in ("bsd68", "imagenet", "hanzi"):
        loader = DataLoader(
            dataset,
            batch_size=1,  # todo customize
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
    elif experiment == "ssi":
        loader = DataLoader(
            dataset,
            batch_size=1,  # todo customize
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
    if experiment in ("bsd68", "hanzi"):
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

    # Calculate scores
    if experiment in ("bsd68",):
        scores = [
            util.calculate_scores(gtx, pred, data_range=255)
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

    result = scores
    # Save results
    scores = pd.DataFrame(scores)
    scores.to_csv("scores.csv")
    np.savez("predictions.npz", **predictions)

    # Show summary
    print("\nEvaluation results:")
    if experiment in ("planaria",):
        pprint(scores.groupby("c").mean())
    else:
        pprint(scores.mean())
    return result


def main(train_dir: str, checkpoint: str = 'last', other_args: list = None) -> None:

    cfg = OmegaConf.load(f'{train_dir}/.hydra/config.yaml')
    if other_args is not None:
        cfg.merge_with_dotlist(other_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"

    print(f"Evaluate backbone {cfg.backbone_name} on experiment {cfg.experiment}, work in {os.getcwd()}")

    cwd = Path(os.getcwd())

    dataset, ground_truth = None, None
    if cfg.experiment not in ("planaria",):
        # For some datasets we need custom loading
        dataset, ground_truth = get_test_dataset_and_gt(cfg, cwd)

    backbone, head = parametrize_backbone_and_head(cfg)

    mdl = model.Noise2Same(
        n_dim=cfg.data.n_dim,
        in_channels=cfg.data.n_channels,
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
    evaluate(evaluator, dataset, ground_truth, cfg.experiment, cfg.training.num_workers, cwd, half)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_dir", required=True,
                        help="Path to hydra train directory")
    parser.add_argument("--checkpoint", choices=["last", "best"],
                        default="last", help="The checkpoint to evaluate, 'last' or 'best'")
    args, unknown_args = parser.parse_known_args()
    main(args.train_dir, args.checkpoint, unknown_args)
