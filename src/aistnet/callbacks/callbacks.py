"""Callback base"""

from typing import Callable


def linear_leaning_rate_reducer(
    start_learn_rate: float,
    end_learn_rate: float,
    start_epoch: int,
    end_epoch: int,
    reduce_until: float,
) -> Callable[[int, float], float]:
    """
    Linear reducer for learn rate based on the given ranges for the learn
    rate and epochs and the percentage information
    until which epoch the learn rate should be reduced

    :param start_learn_rate: initial learn rate
    :param end_learn_rate: target learn rate
    :param start_epoch: start epoch if learning continued
    :param end_epoch: end epoch for the current run
    :param reduce_until: percentage value until which epoch the learn rate
        should be reduce to
    :return: scheduler function for the tf.keras.callbacks.LearningRateScheduler
    """
    if reduce_until > 1 or reduce_until < 0:
        raise ValueError("reduce until must be in the range of [0, 1]")
    epochs_for_reduce = (end_epoch - start_epoch) * reduce_until
    target_epoch = epochs_for_reduce + start_epoch
    reducer = (start_learn_rate - end_learn_rate) // epochs_for_reduce

    def scheduler(epoch: int, lr: float) -> float:
        if epoch < target_epoch:
            return lr - reducer
        return end_learn_rate

    return scheduler
