from functools import partial

import pytest

from noise2same.denoiser.blind2unblind import Blind2Unblind, schedule_lambda_rev
from noise2same.scheduler import HyperparameterScheduler


@pytest.mark.parametrize("total_steps", [100, 200])
@pytest.mark.parametrize("step_threshold", [0.4, 0.8])
def test_hyperparameter_scheduler(total_steps: int, step_threshold: float):
    model = Blind2Unblind(mask_window_size=2, lambda_rev=2)
    schedule_fn = partial(schedule_lambda_rev,
                          lambda_rev=2,
                          lambda_rev_max=20,
                          step_threshold=step_threshold,
                          total_steps=total_steps,
                          )
    hyperparameter_scheduler = HyperparameterScheduler(
        model,
        lambda_rev=schedule_fn
    )

    for i in range(total_steps):
        assert model.lambda_rev == schedule_fn(i)
        hyperparameter_scheduler.step()
