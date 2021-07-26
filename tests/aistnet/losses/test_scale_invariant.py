import numpy as np
import tensorflow as tf

from aistnet.losses.regression import ScaleInvariantLoss, scale_invariant_loss


def test_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    sil = ScaleInvariantLoss(np.asarray(y_true))
    # when
    res = sil(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()
    # then
    assert res == np.float32(0.375)


def test_inner_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    # when
    res = scale_invariant_loss(np.asarray(y_true))(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    ).numpy()
    # then
    assert res == np.float32(0.375)
