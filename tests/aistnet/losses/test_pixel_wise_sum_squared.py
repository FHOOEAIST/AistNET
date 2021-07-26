import numpy as np
import tensorflow as tf

from aistnet.losses.regression import PixelWiseSumSquared, pixel_wise_sum_squared


def test_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    pws = PixelWiseSumSquared()
    # when
    res = pws(y_true, y_pred).numpy()
    # then
    assert res == np.float32(2.0)


def test_inner_base_result():
    # given
    y_true = tf.convert_to_tensor([[0.0, 1.0], [0.0, 0.0]])
    y_pred = tf.convert_to_tensor([[1.0, 1.0], [1.0, 0.0]])
    # when
    res = pixel_wise_sum_squared(y_true, y_pred).numpy()
    # then
    assert res == np.float32(2.0)
