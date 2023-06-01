import torch


def exponential_decay(
        decay_rate: float = 0.5, decay_steps: int = 5e3, staircase: bool = True
):
    """
    Lambda for torch.optimizers.lr_scheduler.LambdaLR mimicking tf.train.exponential_decay:
    decayed_learning_rate = learning_rate *
                            decay_rate ^ (global_step / decay_steps)

    :param decay_rate: float, multiplication factor
    :param decay_steps: int, how many steps to make to multiply by decay_rate
    :param staircase: bool, integer division global_step / decay_steps
    :return: lambda(epoch)
    """

    def _lambda(epoch: int):
        exp = epoch / decay_steps
        if staircase:
            exp = int(exp)
        return decay_rate ** exp

    return _lambda


class ExponentialDecayScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, decay_rate: float = 0.5, decay_steps: int = 5e3, staircase: bool = True):
        super().__init__(optimizer,
                         lr_lambda=exponential_decay(
                             decay_rate=decay_rate,
                             decay_steps=decay_steps,
                             staircase=staircase,
                         ))
