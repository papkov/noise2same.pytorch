from typing import Callable, Dict
from torch import nn


class HyperparameterScheduler:
    def __init__(self, model: nn.Module, **scheduling_fns: Callable):
        self.model = model
        self.scheduling_fns = scheduling_fns
        self.scheduling_step = 0
        self.last_values = {}

    def step(self):
        self.scheduling_step += 1
        for name, fn in self.scheduling_fns.items():
            self.last_values[name] = fn(self.scheduling_step)
            setattr(self.model, name, self.last_values[name])

    def get_last_values(self) -> Dict[str, float]:
        return self.last_values
