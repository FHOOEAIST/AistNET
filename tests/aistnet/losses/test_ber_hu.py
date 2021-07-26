import numpy as np
import tensorflow as tf

from aistnet.losses.regression import BerHu, ber_hu_loss


def test_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    bh = BerHu()
    # when
    res = bh(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()
    # then
    assert res == np.float32(1.3)


def test_inner_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    # when
    res = ber_hu_loss(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    ).numpy()
    # then
    assert res == np.float32(1.3)
