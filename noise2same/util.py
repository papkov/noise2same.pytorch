import os
import random
import string
from functools import partial
from typing import Any, Dict, Tuple, Union

import cv2
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from numpy.linalg import norm
from omegaconf import DictConfig
from omegaconf import OmegaConf
from scipy.fft import dct
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
import logging

log = logging.getLogger(__name__)


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


def crop_as_gt(x: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Crops x to gt shape evenly from each side
    (assumes even padding to remove)
    :param x:
    :param gt:
    :return: cropped x
    """
    return crop_as(x, gt.shape)


def crop_as(x: np.ndarray, target_shape: np.shape) -> np.ndarray:
    """
    Crops x to gt shape evenly from each side
    (assumes even padding to remove)
    :param x:
    :param target_shape:
    :return: cropped x
    """
    try:
        diff = np.array(x.shape) - np.array(target_shape)
    except ValueError:
        raise ValueError(
            f"Shapes do not match: {x.shape} and {target_shape}"
        )
    assert np.all(diff >= 0), f"Target shape {target_shape} is larger than {x.shape} by {diff}"
    top_left = diff // 2
    bottom_right = diff - top_left
    sl = tuple(slice(tl, s - br) for tl, s, br in zip(top_left, x.shape, bottom_right))
    crop = x[sl]
    assert np.all(np.array(crop.shape) == np.array(target_shape))
    return crop


def center_crop(x: np.ndarray, size: int = 63) -> np.ndarray:
    """
    Crops a central part of an array
    (used for PSF)
    :param x: source
    :param size: to crop
    :return: cropped array
    """
    h = size // 2
    return x[tuple(slice(max(0, d // 2 - h), min(d // 2 + h + 1, d)) for d in x.shape)]


def calculate_scores(
        gt: np.ndarray,
        x: np.ndarray,
        data_range: float = 1.0,
        normalize_pairs: bool = False,
        scale: bool = False,
        clip: bool = False,
        multichannel: bool = False,
        prefix: str = "",
        metrics: Tuple[str, ...] = ("rmse", "psnr", "ssim"),
        **kwargs: Any,
) -> Dict[str, float]:
    """
    Calculates image reconstruction metrics
    :param gt: ndarray, the ground truth image
    :param x: ndarray, prediction
    :param data_range: The data range of the input image, 1 by default (0-1 normalized images)
    :param normalize_pairs: bool, normalize and affinely scale pairs gt-x (needed for Planaria dataset)
    :param scale: bool, scale images by min and max (needed for Imagenet dataset)
    :param clip: bool, clip an image to [0, data_range]
    :param multichannel: If True, treat the last dimension of the array as channels for SSIM. Similarity
        calculations are done independently for each channel then averaged.
    :param prefix: str, prefix for metric names
    :param metrics: tuple of metrics to calculate
    :param kwargs: kwargs for SSIM
    :return:
    """

    metrics_fn = {
        'rmse': lambda a, b: np.sqrt(mean_squared_error(a, b)),
        'psnr': partial(peak_signal_noise_ratio, data_range=data_range),
        'ssim': partial(structural_similarity, data_range=data_range, multichannel=multichannel, **kwargs),
        'mi': mutual_information,
        'smi': spectral_mutual_information,
    }

    for metric in metrics:
        assert metric in metrics_fn, f"Unknown metric {metric}"

    x_ = crop_as_gt(x, gt)
    assert gt.shape == x_.shape, f"Different shapes {gt.shape}, {x_.shape}"
    if scale:
        x_ = normalize_zero_one(x_) * data_range
    if normalize_pairs:
        gt, x_ = normalize_min_mse(gt, x_)
    if clip:
        x_ = np.clip(x_, 0, data_range)

    if prefix:
        prefix += "."

    scores = {prefix + metric: metrics_fn[metric](gt, x_) for metric in metrics}
    return scores


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


def normalize_percentile(
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


normalize_zero_one = partial(normalize_percentile, p_min=0, p_max=100, clip=True)


def normalize_zero_one_dict(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalizes all images in the given dictionary to the range [0, 1].
    """
    return {k: normalize_zero_one(v) for k, v in images.items()}


def normalize_min_mse(gt: np.ndarray, x: np.ndarray, normalize_gt: bool = True):
    """
    Normalizes and affinely scales an image pair such that the MSE is minimized
    :param gt: ndarray, the ground truth image
    :param x: ndarray, the image that will be affinely scaled
    :param normalize_gt: bool, set to True of gt image should be normalized (default)
    :return: gt_scaled, x_scaled
    """
    if normalize_gt:
        gt = normalize_percentile(gt, 0.1, 99.9, clip=False).astype(
            np.float32, copy=False
        )
    x = x.astype(np.float32, copy=False) - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x


def plot_3d(im: ndarray) -> None:
    """
    Plot 3D image as three max projections
    :param im: image to plot
    :return: none
    """
    fig = plt.figure(constrained_layout=False, figsize=(12, 7))
    gs = fig.add_gridspec(nrows=3, ncols=5)

    ax_0 = fig.add_subplot(gs[:-1, :-1])
    ax_0.imshow(np.max(im, 0))

    ax_1 = fig.add_subplot(gs[-1, :-1])
    ax_1.imshow(np.max(im, 1))

    ax_2 = fig.add_subplot(gs[:-1, -1])
    ax_2.imshow(np.rot90(np.max(im, 2)))

    plt.setp([ax_0, ax_1, ax_2], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


def concat_projections(im: ndarray, axis: int = 1) -> ndarray:
    """
    Do max projection of an image to all axes and concatenate them in 2D image
    Expects image to be a cube
    :param im: ND image
    :param axis: concatenate projections along it (0 - vertical concatenation, 1 - horizontal)
    :return: 2D concatenation of max projections
    """
    projections = []
    for i in range(im.ndim):
        p = np.max(im, axis=i)
        if i > 0 and axis > 0:
            p = np.rot90(p)
        projections.append(p)
    projections = np.concatenate(projections, axis=axis)
    return projections


def concat_projections_3d(im: ndarray, projection_func: callable = np.max) -> ndarray:
    """
    Do max projection of an image to all axes and concatenate them in 2D image
    :param im: ND image
    :param projection_func: function to make 2d from 3d, np.max by default
    :return: 2D concatenation of max projections
    """
    projections = np.zeros((im.shape[0] + im.shape[1], im.shape[0] + im.shape[2]))
    shifts = [(0, 0), (im.shape[1], 0), (0, im.shape[2])]
    for i, s in enumerate(im.shape):
        p = projection_func(im, axis=i)
        if i == 2:
            p = np.rot90(p)
        ps = tuple(slice(0 + sh, d + sh) for d, sh in zip(p.shape, shifts[i]))
        projections[ps] = p
    return projections


def plot_projections(im: ndarray, axis: int = 1) -> None:
    """
    Plot batch projections from `concat_projections`
    :param im: ND image
    :param axis: concatenate projections along it (0 - vertical concatenation, 1 - horizontal)
    :return:
    """
    projections = concat_projections(im, axis)
    fig, ax = plt.subplots()
    ax.imshow(projections)
    clean_plot(ax)


def load_checkpoint_to_module(module, checkpoint_path: str):
    """
    Loads PyTorch state checkpoint to module
    :param module: nn.Module
    :param checkpoint_path: str, path to checkpoint
    :return:
    """
    checkpoint = torch.load(checkpoint_path)
    for attr, state_dict in checkpoint.items():
        try:
            getattr(module, attr).load_state_dict(state_dict)
        except AttributeError:
            log.warning(
                f"Attribute {attr} is present in the checkpoint but absent in the class, do not load"
            )


def detach_to_np(
        images: Dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    Detaches and denormalizes all tensors in the given dictionary, then converts to np.array.
    """
    return {
        k: np.moveaxis(
            (v * std.to(v.device) + mean.to(v.device)).detach().cpu().numpy(),
            1,
            -1,
        )
        for k, v in images.items()
    }


# Metrics from SSI
def spectral_mutual_information(image_a, image_b, normalised=True):
    norm_image_a = image_a / norm(image_a.flatten(), 2)
    norm_image_b = image_b / norm(image_b.flatten(), 2)

    dct_norm_true_image = dct(dct(norm_image_a, axis=0), axis=1)
    dct_norm_test_image = dct(dct(norm_image_b, axis=0), axis=1)

    return mutual_information(
        dct_norm_true_image, dct_norm_test_image, normalised=normalised
    )


def mutual_information(image_a, image_b, bins=256, normalised=True):
    image_a = image_a.flatten()
    image_b = image_b.flatten()

    c_xy = np.histogram2d(image_a, image_b, bins)[0]
    mi = mutual_info_from_contingency(c_xy)
    mi = mi / joint_entropy_from_contingency(c_xy) if normalised else mi
    return mi


def joint_entropy_from_contingency(contingency):
    # coordinates of non-zero entries in contingency table:
    nzx, nzy = np.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingency table:
    contingency_sum = contingency.sum()

    # normalised contingency, i.e. probability:
    p = nz_val / contingency_sum

    # log contingency:
    log_p = np.log2(p)

    # Joint entropy:
    joint_entropy = -p * log_p

    return joint_entropy.sum()


def mutual_info_from_contingency(contingency):
    # cordinates of non-zero entries in contingency table:
    nzx, nzy = np.nonzero(contingency)

    # non zero values:
    nz_val = contingency[nzx, nzy]

    # sum of all values in contingnecy table:
    contingency_sum = contingency.sum()

    # marginals:
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    #
    log_contingency_nm = np.log2(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log2(outer) + np.log2(pi.sum()) + np.log2(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - np.log2(contingency_sum))
        + contingency_nm * log_outer
    )
    return mi.sum()


def ssim(prediction, target):
    """
    https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/2fff2978/train.py#L258
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    """
    Calculate SSIM
    the same outputs as MATLAB's
    https://github.com/TaoHuang2018/Neighbor2Neighbor/blob/2fff2978/train.py#L279
    img1, img2: [0, 255]
    """
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def flatten_config(cfg: DictConfig) -> Dict:
    """
    Flattens the config to a dictionary for logging
    :param cfg: hydra config
    :return: dict with flattened config
    """
    d_cfg = {}
    for group, group_dict in dict(cfg).items():
        if isinstance(group_dict, DictConfig):
            for param, value in dict(group_dict).items():
                d_cfg[f"{group}.{param}"] = value
        else:
            d_cfg[group] = group_dict
    return d_cfg


def register_config_resolvers():
    """
    Registers custom resolvers for OmegaConf for config interpolation
    :return:
    """
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('max', max)
    OmegaConf.register_new_resolver('ceil', lambda x: math.ceil(eval(x)))
    OmegaConf.register_new_resolver('suffix', lambda: ''.join(random.choices(string.ascii_uppercase, k=5)))
