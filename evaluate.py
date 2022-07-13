import glob
import os
from pathlib import Path
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from noise2same import model, util
from noise2same.dataset.getter import (
    get_planaria_dataset_and_gt,
    get_test_dataset_and_gt,
)
from noise2same.evaluator import Evaluator
from utils import parametrize_backbone_and_head


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    if "backbone_name" not in cfg.keys():
        print("Please specify a backbone with `+backbone=name`")
        return

    if "experiment" not in cfg.keys():
        print("Please specify an experiment with `+experiment=name`")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.device}"

    cwd = Path(get_original_cwd())
    print(f"Evaluate backbone {cfg.backbone_name} on experiment {cfg.experiment}, work in {os.getcwd()}")

    dataset, ground_truth = None, None
    if cfg.experiment not in ("planaria",):
        # For some datasets we need custom loading
        dataset, ground_truth = get_test_dataset_and_gt(cfg)

    loader = None
    if cfg.experiment.lower() in ("bsd68", "imagenet", "hanzi"):
        loader = DataLoader(
            dataset,
            batch_size=1,  # todo customize
            num_workers=cfg.training.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

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
    print(os.getcwd())
    checkpoint_path = (
        cfg.checkpoint
        if hasattr(cfg, "checkpoint")
        else cwd / f"weights/{cfg.experiment}.pth"
    )

    # Run evaluation
    half = getattr(cfg, "amp", False)
    masked = getattr(cfg, "masked", False)
    evaluator = Evaluator(mdl, checkpoint_path=checkpoint_path, masked=masked)
    if cfg.experiment in ("bsd68", "hanzi", "imagenet"):
        predictions = evaluator.inference(loader, half=half)
    elif cfg.experiment in ("microtubules",):
        predictions = evaluator.inference_single_image_dataset(
            dataset, half=half, batch_size=1
        )
    elif cfg.experiment in ("planaria",):
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
    if cfg.experiment not in ("planaria", "microtubules"):
        predictions = {k: [d[k].squeeze() for d in predictions] for k in predictions[0]}

    # Calculate scores
    if cfg.experiment in ("bsd68",):
        scores = [
            util.calculate_scores(gtx, pred, data_range=255)
            for gtx, pred in zip(ground_truth, predictions["image"])
        ]
    elif cfg.experiment in ("hanzi",):
        scores = [
            util.calculate_scores(gtx * 255, pred, data_range=255, scale=True)
            for gtx, pred in zip(ground_truth, predictions["image"])
        ]
    elif cfg.experiment in ("imagenet",):
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
    elif cfg.experiment in ("microtubules",):
        scores = util.calculate_scores(ground_truth, predictions, normalize_pairs=True)
    elif cfg.experiment in ("planaria",):
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

    # Save results
    scores = pd.DataFrame(scores)
    scores.to_csv("scores.csv")
    np.savez("predictions.npz", **predictions)

    # Show summary
    print("\nEvaluation results:")
    if cfg.experiment in ("planaria",):
        pprint(scores.groupby("c").mean())
    else:
        pprint(scores.mean())


if __name__ == "__main__":
    main()
