import numpy as np
import tensorflow as tf

from aistnet.losses.regression import SumOfSquareDifference, sum_of_difference_square


def test_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    ssd = SumOfSquareDifference()
    # when
    res = ssd(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()
    # then
    assert res == np.float32(2.0)


def test_inner_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    # when
    res = sum_of_difference_square(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    ).numpy()
    # then
    assert res == np.float32(2.0)


def test_inner_base_result_2():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[2.0, 1.0], [2.0, 0.0]]
    # when
    res = sum_of_difference_square(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    ).numpy()
    # then
    assert res == np.float32(8.0)
