import numpy as np
import tensorflow as tf

from aistnet.losses.regression import MeanDifference, mse_scaled


def test_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    md = MeanDifference()
    # when
    res = md(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()
    # then
    assert res == np.float32(-0.5)


def test_inner_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    # when
    res = mse_scaled(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()
    # then
    assert res == np.float32(-0.5)
