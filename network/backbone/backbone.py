from torch import nn
from abc import ABC, abstractmethod


class AbstractBackbone(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
