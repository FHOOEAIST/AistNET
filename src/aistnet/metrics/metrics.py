"""Custom Metric functions"""

import tensorflow as tf


def mean_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate mean iou (intersection over union)

    :param y_true: target tensor with the expected result
    :param y_pred: actual result form the model
    :return: float as metric value
    """
    y_true = tf.cast(y_true, tf.dtypes.float64)
    y_pred = tf.cast(y_pred, tf.dtypes.float64)
    i = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    u = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - i
    return tf.reduce_mean(i / u)


def dice_coefficient(
    y_true: tf.Tensor, y_pred: tf.Tensor, smooth: int = 1
) -> tf.Tensor:
    """
    Calculate the dice coefficient

    :param y_true: target tensor with the expected result
    :param y_pred: actual result form the model
    :param smooth: smooth value for intersection over union
    :return: float as metric value
    """
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(
        y_pred, axis=[1, 2, 3]
    )
    dice = tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    return dice
