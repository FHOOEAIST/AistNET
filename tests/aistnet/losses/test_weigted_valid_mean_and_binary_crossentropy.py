import numpy as np

from aistnet.losses.classification import (
    WeightedValidMeanAndBinaryCrossentropy,
    pixel_wise_weighted_valid_mean_and_binary_crossentropy,
)


def test_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.2]]
    y_pred = [[1.0, 1.0], [1.0, 0.3]]
    wmdc = WeightedValidMeanAndBinaryCrossentropy()
    # when
    res = wmdc(y_true, y_pred).numpy()
    # then
    assert res == np.float32(3.6365767)


def test_inner_base_result():
    # given
    y_true = [[0.0, 1.0], [0.0, 0.2]]
    y_pred = [[1.0, 1.0], [1.0, 0.3]]
    # when
    res = pixel_wise_weighted_valid_mean_and_binary_crossentropy()(
        y_true, y_pred
    ).numpy()
    # then
    assert np.all(
        res
        == np.asarray(
            [[7.4041195e00, -2.6249999e-01], [7.4041195e00, 5.6716800e-04]]
        ).astype(dtype=np.float32)
    )
