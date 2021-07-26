"""Regression Loss Functions"""

from functools import reduce
from typing import Any, Callable, Dict

import numpy as np
import tensorflow as tf

from aistnet.losses.AistNetLoss import AistNetLoss


class PixelWiseSumSquared(AistNetLoss):
    """
    PixelWiseSumSquared loss function class
    """

    def __init__(self, name: str = "pixel_wise_sum_squared", **kwargs: Dict[str, Any]):
        """
        Computes the sum of squared difference between the labels and
        predictions on the last level such as pixel level for example.

        `loss = sum(square(y_true - y_pred))`

        Standalone usage:

        >>> y_true = [[0., 1.], [0., 0.]]
        >>> y_pred = [[1., 1.], [1., 0.]]
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> pws = aistnet.losses.PixelWiseSumSquared()
        >>> pws(y_true, y_pred).numpy()
        >>> 2.
        """
        super().__init__(pixel_wise_sum_squared, name, **kwargs)


def pixel_wise_sum_squared(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the sum of squared difference between the label and
    the predictions on the last level such as pixel level for example.

    :param y_true: target tensor with the expected result
    :param y_pred: actual result form the model
    :return: float tensor as loss value
    """
    squared_diff = tf.math.squared_difference(y_true, y_pred)
    summed_squared_diff = tf.keras.backend.sum(squared_diff)
    return summed_squared_diff


class MeanDifference(AistNetLoss):
    """
    MeanDifference loss function class
    """

    def __init__(self, name: str = "mean_difference", **kwargs: Dict[str, Any]):
        """
        Computes the mean difference between the label and the predictions
        on the last level such as pixel level for example.

        `loss = mean(y_true - y_pred)`

        >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]])
        >>> y_pred = tf.convert_to_tensor([[1., 1.], [1., 0.]])
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> md = aistnet.losses.MeanDifference()
        >>> md(y_true, y_pred).numpy()
        >>> -0.5
        """
        super().__init__(mse_scaled, name, **kwargs)


def mse_scaled(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Mean difference between the label and the predictions
    on the last level such as pixel level for example.

    :param y_true: target tensor with the expected result
    :param y_pred: actual result form the model
    :return: float tensor as loss value
    """
    return tf.math.reduce_mean(y_true - y_pred)


class SumOfSquareDifference(AistNetLoss):
    """
    SumOfSquareDifference loss function class
    """

    def __init__(
        self, name: str = "sum_of_difference_square", **kwargs: Dict[str, Any]
    ):
        """
        Computes the sum of square differences between the label and the
        predictions on the last level such as pixel level for example.

        `loss = sum((y_true - y_pred)^2)`

        >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]])
        >>> y_pred = tf.convert_to_tensor([[1., 1.], [1., 0.]])
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> ssd = aistnet.losses.SumOfSquareDifference()
        >>> ssd(y_true, y_pred).numpy()
        >>> 2.
        """
        super().__init__(sum_of_difference_square, name, **kwargs)


def sum_of_difference_square(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the sum of square differences between the label and the
    predictions on the last level such as pixel level for example.

    :param y_true: target tensor with the expected result
    :param y_pred: actual result form the model
    :return: float as loss value
    """
    squared_diff = tf.keras.backend.square(y_true - y_pred)
    summed_squared_diff = tf.keras.backend.sum(squared_diff)
    return summed_squared_diff


class SumOfAbsoluteDifference(AistNetLoss):
    """
    SumOfAbsoluteDifference loss function class
    """

    def __init__(
        self, name: str = "sum_of_absolute_difference", **kwargs: Dict[str, Any]
    ):
        """
        Computes the sum difference between the label and
        the predictions on the last level such as pixel level for example.

        `loss = sum(abs((y_true - y_pred)))`

        >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]])
        >>> y_pred = tf.convert_to_tensor([[1., 1.], [1., 0.]])
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> ssd = aistnet.losses.SumOfAbsoluteDifference()
        >>> ssd(y_true, y_pred).numpy()
        >>> 2.
        """
        super().__init__(sum_of_absolute_square, name, **kwargs)


def sum_of_absolute_square(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the sum difference between the label and
    the predictions on the last level such as pixel level for example.

    :param y_true: target tensor with the expected result
    :param y_pred: actual result form the model
    :return: float tensor as loss value
    """
    abs_diff = tf.math.abs(y_true - y_pred)
    summed_diff = tf.keras.backend.sum(abs_diff)
    return summed_diff


class BerHu(AistNetLoss):
    """
    BerHu loss function class
    """

    def __init__(self, name: str = "ber_hu", **kwargs: Dict[str, Any]):
        """
        Computes the BerHu difference between the label and the
        predictions on the last level such as pixel level for example.

        https://arxiv.org/abs/1207.6868

        >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]])
        >>> y_pred = tf.convert_to_tensor([[1., 1.], [1., 0.]])
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> bh = aistnet.losses.BerHu()
        >>> bh(y_true, y_pred).numpy()
        >>> 1.3
        """
        super().__init__(ber_hu_loss, name, **kwargs)


def ber_hu_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the BerHu difference between the label and the
    predictions on the last level such as pixel level for example.

    https://arxiv.org/abs/1207.6868

    :param y_true: target tensor with the expected result
    :param y_pred: actual result form the model
    :return: float as loss value
    """
    abs_error = tf.math.abs(y_true - y_pred)
    delta = 0.2 * tf.math.reduce_max(abs_error)

    l2 = (tf.math.multiply(abs_error, abs_error) / delta + delta) * 0.5

    mask_down_f = tf.cast(tf.math.less(abs_error, delta), tf.float32)
    mask_up_f = tf.cast(tf.math.greater(abs_error, delta), tf.float32)

    loss = abs_error * mask_down_f + l2 * mask_up_f

    return tf.keras.backend.mean(loss)


class ScaleInvariantLoss(AistNetLoss):
    """
    ScaleInvariantLoss loss function class
    """

    def __init__(
        self,
        target: np.ndarray,
        name: str = "scale_invariant",
        **kwargs: Dict[str, Any]
    ):
        """
        Computes the scale invariant difference between the label and the
        predictions on the last level such as pixel level for example.

        https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf

        `loss = 1/n * Σ d^2 - λ/n^2 * (Σ d)^2 | d = log y_pred - log y_true`

        >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]])
        >>> y_pred = tf.convert_to_tensor([[1., 1.], [1., 0.]])
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> sil = aistnet.losses.ScaleInvariantLoss(y_true.numpy())
        >>> sil(y_true, y_pred).numpy()
        >>> 0.375

        :param target: a target sample to retrieve the dimension size
        """
        super().__init__(scale_invariant_loss(target), name, **kwargs)


def scale_invariant_loss(
    target: np.ndarray,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Computes the scale invariant difference between the label and the
    predictions on the last level such as pixel level for example.

    https://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf

    :param target: a target sample to retrieve the dimension size
    :return: callable parameterized loss function
    """
    pixels = reduce(lambda x, y: x * y, target.shape[:2])
    first_factor = 1 / pixels  # 1/n
    second_factor = 0.5 / (pixels * pixels)  # lambda/n^2

    def _scale_invariant_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # log_actual = tf.math.log(y_true)
        # log_pred = tf.math.log(y_pred)
        first = first_factor * tf.math.reduce_sum(
            tf.math.square(y_pred - y_true)
        )  # 1/n * Σ d^2
        second = second_factor * tf.math.square(
            tf.math.reduce_sum(y_pred - y_true)
        )  # λ/n^2 * (Σ d)^2
        return first - second

    return _scale_invariant_loss
