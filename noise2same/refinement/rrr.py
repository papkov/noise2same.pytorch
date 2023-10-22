import torch.nn as nn
from torch import Tensor as T
import torch


class RandomReplacementRefinement(nn.Module):

    def __init__(self, num_iter: int = 8, p: float = 0.16):
        super().__init__()
        self.num_iter = num_iter
        self.p = p

    def forward(self, model: nn.Module, x: T, model_out: T):
        result = torch.empty(self.num_iter, *x.shape)
        for i in range(self.num_iter):
            mask = torch.rand_like(x) < self.p
            in_ = torch.clone(model_out)
            in_[mask] = x[mask]
            result[i] = model(in_)
        return result.mean(dim=0)
