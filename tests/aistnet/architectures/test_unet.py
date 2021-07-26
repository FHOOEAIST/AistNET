import pytest
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model

from aistnet.architectures.unet import cnn_2d_auto_encoder_with_skip


@pytest.fixture(autouse=True)
def clean_tensorflow_context():
    # clean tensorflow session before each test
    clear_session()


def test_base_with_one_encoder():
    # given
    dims = [240, 224, 1]
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(1)(dims)
    # then
    assert in_.get_shape().as_list() == [None, *dims]
    assert in_.name == "input_1"
    assert out_.name == "conv2d_8/Relu:0"
    assert out_.get_shape().as_list() == in_.get_shape().as_list()


def test_base_with_one_encoder_and_model():
    # given
    dims = [240, 224, 1]
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(1)(dims)
    model = Model(inputs=[in_], outputs=[out_])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # then
    assert len(model.layers) == 23
    assert model.input.get_shape().as_list() == [None, *dims]
    assert model.layers[0].name == str(in_.name).split(":")[0]
    assert model.count_params() == 26181
    assert model.output.get_shape().as_list() == [None, *dims]


def test_base_with_two_encoder():
    # gives
    dims = [240, 224, 1]
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(2, final_activation="sigmoid")(dims)
    # then
    assert in_.get_shape().as_list() == [None, *dims]
    assert in_.name == "input_1"
    assert out_.name == "conv2d_13/Sigmoid:0"
    assert out_.get_shape().as_list() == in_.get_shape().as_list()


def test_base_with_two_encoder_and_model():
    # given
    dims = [240, 224, 1]
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(2)(dims)
    model = Model(inputs=[in_], outputs=[out_])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # then
    assert len(model.layers) == 37
    assert model.input.get_shape().as_list() == [None, *dims]
    assert model.layers[0].name == str(in_.name).split(":")[0]
    assert model.count_params() == 118309
    assert model.output.get_shape().as_list() == [None, *dims]


def test_base_with_one_encoder_and_model_no_normalize():
    # given
    dims = [240, 224, 1]
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(
        1, normalize_encoder=False, normalize_bottom=False, normalize_decoder=False
    )(dims)
    model = Model(inputs=[in_], outputs=[out_])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # then
    assert len(model.layers) == 17
    assert model.input.get_shape().as_list() == [None, *dims]
    assert model.layers[0].name == str(in_.name).split(":")[0]
    assert model.count_params() == 25669
    assert model.output.get_shape().as_list() == [None, *dims]


def test_base_with_two_encoder_and_model_no_normalize():
    # given
    dims = [240, 224, 1]
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(
        2, normalize_encoder=False, normalize_bottom=False, normalize_decoder=False
    )(dims)
    model = Model(inputs=[in_], outputs=[out_])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # then
    assert len(model.layers) == 27
    assert model.input.get_shape().as_list() == [None, *dims]
    assert model.layers[0].name == str(in_.name).split(":")[0]
    assert model.count_params() == 117029
    assert model.output.get_shape().as_list() == [None, *dims]


def test_base_with_one_encoder_and_model_no_max_pooling():
    # given
    dims = [240, 224, 1]
    drop = 0.5
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(1, drop_factor=drop)(dims)
    model = Model(inputs=[in_], outputs=[out_])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # then
    assert len(model.layers) == 23
    assert model.input.get_shape().as_list() == [None, *dims]
    assert model.layers[0].name == str(in_.name).split(":")[0]
    assert model.count_params() == 26181
    assert model.output.get_shape().as_list() == [None, *dims]


def test_base_with_two_encoder_and_model_no_max_pooling():
    # given
    dims = [240, 224, 1]
    drop = 0.5
    # when
    in_, out_ = cnn_2d_auto_encoder_with_skip(2, drop_factor=drop)(dims)
    model = Model(inputs=[in_], outputs=[out_])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # then
    assert len(model.layers) == 37
    assert model.input.get_shape().as_list() == [None, *dims]
    assert model.layers[0].name == str(in_.name).split(":")[0]
    assert model.count_params() == 118309
    assert model.output.get_shape().as_list() == [None, *dims]
