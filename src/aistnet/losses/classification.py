"""Classification Loss Functions"""
from typing import Any, Callable, Dict

import tensorflow as tf

from aistnet.losses.AistNetLoss import AistNetLoss


class WeightedMeanAndBinaryCrossentropy(AistNetLoss):
    """
    WeightedMeanAndBinaryCrossentropy loss function class
    """

    def __init__(
        self,
        ratio: float = 0.5,
        name: str = "mean_of_binary_crossentropy",
        **kwargs: Dict[str, Any]
    ):
        """
        Computes a weighted mean difference and a weighted binary crossentropy
        between the  labels and predictions

        `loss = ??`

        Standalone usage:

        >>> y_true = [[0., 1.], [0., 0.]]
        >>> y_pred = [[1., 1.], [1., 0.]]
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> pws = aistnet.losses.WeightedMeanAndBinaryCrossentropy()
        >>> pws(y_true, y_pred).numpy()
        >>> 4.0833097

        :param ratio: value to balance between binary crossentropy (1 - ratio)
            and mean difference (ratio) error
        """
        super().__init__(weighted_mean_and_binary_crossentropy(ratio), name, **kwargs)


def weighted_mean_and_binary_crossentropy(
    ratio: float = 0.5,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Computes a weighted mean difference and a weighted binary crossentropy
    between the  labels and predictions

    :param ratio: value to balance between binary crossentropy (1 - ratio)
            and mean difference (ratio) error
    :return: callable parameterized loss function
    """

    cross_weight = 1 - ratio
    mean_weight = ratio

    def _call(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes a weighted mean difference and a weighted binary crossentropy
        between the  labels and predictions

        :param y_true: target tensor with the expected result
        :param y_pred: actual result form the model
        :return: float as loss value
        """
        res_bce = tf.keras.backend.binary_crossentropy(
            y_true, y_pred
        )  # working with orig
        res_m = tf.keras.losses.mean_absolute_error(y_true, y_pred)  # working with orig
        return res_bce * cross_weight + res_m * mean_weight

    return _call


class WeightedValidMeanAndBinaryCrossentropy(AistNetLoss):
    """
    WeightedValidMeanAndBinaryCrossentropy loss function class
    """

    def __init__(
        self,
        ratio: float = 0.5,
        name: str = "pixel_wise_valid_mean_squared",
        **kwargs: Dict[str, Any]
    ):
        """
        Computes a masked weighted pixel-wise binary cross entropy with weighted
        mean difference between the label and the predictions

        `loss = ??`

        >>> y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]])
        >>> y_pred = tf.convert_to_tensor([[1., 1.], [1., 0.]])
        >>> # Using 'auto'/'sum_over_batch_size' reduction type.
        >>> pms = aistnet.losses.WeightedValidMeanAndBinaryCrossentropy()
        >>> pms(y_true, y_pred).numpy()
        >>> 3.6365767

        :param ratio: value to balance between binary crossentropy (1 - ratio)
            and mean difference (ratio) error
        """
        super().__init__(
            pixel_wise_weighted_valid_mean_and_binary_crossentropy(ratio),
            name,
            **kwargs
        )


def pixel_wise_weighted_valid_mean_and_binary_crossentropy(
    ratio: float = 0.5,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Computes a masked weighted pixel-wise binary cross entropy with weighted
        mean difference between the label and the predictions

    :param ratio: value to balance between binary crossentropy (1 - ratio)
        and mean difference (ratio) error
    :return: callable parameterized loss function
    """
    cross_weight = 1 - ratio
    mean_weight = ratio

    def _call(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes a masked weighted pixel-wise binary cross entropy with weighted
        mean difference between the label and the predictions.

        :param y_true: target tensor with the expected result
        :param y_pred: actual result form the model
        :return: float as loss value
        """
        mask = tf.where(y_true == 1.0, 0.0, 1.0)
        y_actual_masked = tf.math.multiply(y_true, mask)
        y_pred_mask = tf.math.multiply(y_pred, mask)
        res_bce = tf.keras.backend.binary_crossentropy(
            y_actual_masked, y_pred_mask
        )  # working with orig
        res_m = tf.math.reduce_mean(y_actual_masked - y_pred_mask)  # working with orig
        return res_bce * cross_weight + res_m * mean_weight

    return _call
