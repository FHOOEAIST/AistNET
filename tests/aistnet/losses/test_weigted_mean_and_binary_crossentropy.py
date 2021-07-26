import numpy as np

from aistnet.losses.classification import (
    WeightedMeanAndBinaryCrossentropy,
    weighted_mean_and_binary_crossentropy,
)


def test_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    wmdc = WeightedMeanAndBinaryCrossentropy()
    # when
    res = wmdc(y_true, y_pred).numpy()
    # then
    assert res == np.float32(4.0833097)


def test_inner_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.0]]
    y_pred = [[1.0, 1.0], [1.0, 0.0]]
    # when
    res = weighted_mean_and_binary_crossentropy()(y_true, y_pred).numpy()
    # then
    assert np.all(
        res
        == np.asarray([[7.9166193, 0.25000], [7.9166193, 0.25000]]).astype(np.float32)
    )
