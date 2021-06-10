import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def clean_plot(ax):
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
