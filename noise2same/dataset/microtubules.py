import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from noise2same.dataset.abc import AbstractNoiseDataset3DLarge


@dataclass
class MicrotubulesDataset(AbstractNoiseDataset3DLarge):
    path: Union[
        Path, str
    ] = "data/microtubules-simulation"
    input_name: str = "input.tif"
    # input_name: str = "input-generated-poisson-gaussian-2e-4.tif"
