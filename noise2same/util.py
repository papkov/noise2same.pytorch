import os
import random
from typing import Dict, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


def clean_plot(ax: np.ndarray) -> None:
    """
    Plot axes without ticks in tight layout
    :param ax: ndarray of matplotlib axes
    :return:
    """
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


def fix_seed(seed: int = 56) -> None:
    """
    Fix all random seeds for reproducibility
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crop_as(x: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Crops x to gt shape evenly from each side
    (removes needed padding)
    :param x:
    :param gt:
    :return: cropped x
    """
    diff = np.array(x.shape) - np.array(gt.shape)
    assert np.all(diff >= 0)
    top_left = diff // 2
    bottom_right = diff - top_left
    crop = x[
        top_left[0] : x.shape[0] - bottom_right[0],
        top_left[1] : x.shape[1] - bottom_right[1],
    ]
    assert crop.shape == gt.shape
    return crop


def calculate_scores(
    gt: np.ndarray, x: np.ndarray, data_range: float = 1.0, multichannel: bool = False
) -> Dict[str, float]:
    """
    Calculates image reconstruction metrics
    :param gt: ndarray, the ground truth image
    :param x: ndarray, prediction
    :param data_range: The data range of the input image, 1 by default (0-1 normalized images)
    :param multichannel: If True, treat the last dimension of the array as channels for SSIM. Similarity
        calculations are done independently for each channel then averaged.
    :return:
    """
    x_ = crop_as(x, gt)
    assert gt.shape == x_.shape, f"Different shapes {gt.shape}, {x_.shape}"
    gt_, x_ = normalize_min_mse(gt, x_)

    metrics = {
        "rmse": np.sqrt(mean_squared_error(gt_, x_)),
        "psnr": peak_signal_noise_ratio(gt_, x_, data_range=data_range),
        "ssim": structural_similarity(
            gt_, x_, data_range=1.0, multichannel=multichannel
        ),
    }

    return metrics


# Normalization utils from Noise2Void
def normalize_mi_ma(
    x: np.ndarray,
    mi: Union[float, np.ndarray],
    ma: Union[float, np.ndarray],
    clip: bool = False,
    eps: float = 1e-20,
    dtype: type = np.float32,
):
    """

    :param x:
    :param mi:
    :param ma:
    :param clip:
    :param eps:
    :param dtype:
    :return:
    """
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)
    if clip:
        x = np.clip(x, 0, 1)
    return x


def normalize(
    x,
    p_min: float = 2.0,
    p_max: float = 99.8,
    axis: Union[int, Tuple[int, ...]] = None,
    clip: bool = False,
    eps: float = 1e-20,
    dtype: type = np.float32,
):
    """
    Percentile-based image normalization.
    :param x:
    :param p_min:
    :param p_max:
    :param axis:
    :param clip:
    :param eps:
    :param dtype:
    :return:
    """

    mi = np.percentile(x, p_min, axis=axis, keepdims=True)
    ma = np.percentile(x, p_max, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_min_mse(gt: np.ndarray, x: np.ndarray, normalize_gt: bool = True):
    """
    Normalizes and affinely scales an image pair such that the MSE is minimized
    :param gt: ndarray, the ground truth image
    :param x: ndarray, the image that will be affinely scaled
    :param normalize_gt: bool, set to True of gt image should be normalized (default)
    :return: gt_scaled, x_scaled
    """
    if normalize_gt:
        gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False) - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x
