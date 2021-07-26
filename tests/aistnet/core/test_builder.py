# ignore no-untyped-def

from types import FunctionType
from typing import List, Tuple

from tensorflow.keras import layers
from tensorflow.keras.losses import BinaryCrossentropy, Loss
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, Optimizer

from aistnet.core.builder import ModelBuilder


class TestModelBuilder(ModelBuilder):
    pass


class MinimalModelBuilder(ModelBuilder):
    def build(self, dimension: List[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = layers.Input(shape=dimension, name="Minimal")
        out_ = layers.Dense(10)(in_)
        return in_, out_


def test_base_model_builder():
    # when
    TestModelBuilder()


def test_base_dimension_overload_builder():
    # given
    dims = [28, 28]
    # when
    TestModelBuilder(input_dimension=dims)


def test_base_loss_overload_builder():
    # given
    loss = BinaryCrossentropy()
    # when
    TestModelBuilder(loss=loss)


def test_base_method_optimizer_overload_build():
    # given
    optimizer = Adam()
    # when
    TestModelBuilder(optimizer=optimizer)


def test_base_str_optimizer_overload_build():
    # given
    optimizer = "adam"
    # when
    TestModelBuilder(optimizer=optimizer)


def test_base_metrics_overload_build():
    # given
    metrics = [Accuracy()]
    # when
    TestModelBuilder(metrics=metrics)


def test_base_function_model_overload_builder():
    # given
    def _builder() -> Tuple[List[layers.Layer], List[layers.Layer]]:
        return None, None  # type: ignore[return-value]

    # when
    TestModelBuilder(builder=_builder)


def test_base_sequential_model_overload_builder():
    # given
    sequential = Sequential()
    # when
    TestModelBuilder(model=sequential)


def test_minimal_by_extending_builder():
    # given
    dims = (28,)
    optimizer = Adam()
    loss = BinaryCrossentropy()
    builder = MinimalModelBuilder(optimizer=optimizer, loss=loss, dimension=dims)
    # when
    model = builder.finalize()
    # then
    assert model is not None
    assert model.layers[0].name == "Minimal"


def test_minimal_no_optimizer():
    # given
    def _builder_func(dimension: List[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = layers.Input(shape=dimension, name="Start")
        out_ = layers.Dense(10)(in_)
        return in_, out_

    dims = (28,)
    loss = BinaryCrossentropy()
    builder = MinimalModelBuilder(builder=_builder_func, loss=loss, dimension=dims)
    # when
    model = builder.finalize()
    # then
    assert model.optimizer.__class__.__name__ == "Adam"


def test_minimal_str_optimizer():
    # given
    def _builder_func(dimension: List[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = layers.Input(shape=dimension, name="Start")
        out_ = layers.Dense(10)(in_)
        return in_, out_

    dims = (28,)
    optimizer = "adam"
    loss = BinaryCrossentropy()
    builder = ModelBuilder(
        builder=_builder_func, optimizer=optimizer, loss=loss, dimension=dims
    )
    # when
    model = builder.finalize()
    # then
    assert model is not None
    assert model.layers[0].name == "Start"


def test_minimal_no_loss():
    # given
    def _builder_func(dimension: List[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = layers.Input(shape=dimension, name="Start")
        out_ = layers.Dense(10)(in_)
        return in_, out_

    dims = (28,)
    builder = ModelBuilder(builder=_builder_func, dimension=dims)
    # when
    model = builder.finalize()
    # then
    assert model.optimizer.__class__.__name__ == "Adam"
    assert model.loss == "binary_crossentropy"


def test_minimal_by_function_builder():
    # given
    def _builder_func(dimension: List[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = layers.Input(shape=dimension, name="Start")
        out_ = layers.Dense(10)(in_)
        return in_, out_

    dims = (28,)
    optimizer = Adam()
    loss = BinaryCrossentropy()
    builder = ModelBuilder(
        builder=_builder_func, dimension=dims, optimizer=optimizer, loss=loss
    )
    # when
    model = builder.finalize()
    # then
    assert model is not None
    assert model.layers[0].name == "Start"


def test_minimal_by_sequential_builder():
    # given
    model = Sequential([layers.Dense(10, name="Dense"), layers.Dense(10)])
    dims = (28,)
    optimizer = Adam()
    loss = BinaryCrossentropy()
    builder = ModelBuilder(model=model, dimension=dims, optimizer=optimizer, loss=loss)
    # when
    model = builder.finalize()
    # then
    assert model is not None
    assert model.layers[0].name == "Dense"


def test_no_dimension_at_finalize():
    # given
    optimizer = Adam()
    loss = BinaryCrossentropy()
    builder = MinimalModelBuilder(optimizer=optimizer, loss=loss)
    # when
    try:
        builder.finalize()
    # then
    except ValueError:
        return
    assert False


def test_no_builder_at_finalize():
    # given
    dims = (28,)
    optimizer = Adam()
    loss = BinaryCrossentropy()
    builder = ModelBuilder(optimizer=optimizer, loss=loss, dimension=dims)
    # when
    try:
        builder.finalize()
    # then
    except ValueError:
        return
    assert False


def test_dimension_setter_getter():
    # given
    dims = (28,)
    builder = ModelBuilder()
    # when
    builder.dimension = dims
    builder_dim = builder.dimension
    # then
    assert dims == builder_dim
    assert isinstance(builder_dim, tuple)


def test_optimizer_setter_getter():
    # given
    optimizer = Adam()
    builder = ModelBuilder()
    # when
    builder.optimizer = optimizer
    builder_optimizer = builder.optimizer
    # then
    assert optimizer == builder_optimizer
    assert isinstance(builder_optimizer, Optimizer)


def test_loss_setter_getter():
    # given
    loss = BinaryCrossentropy()
    builder = ModelBuilder()
    # when
    builder.loss = loss
    builder_loss = builder.loss
    # then
    assert loss == builder_loss
    assert isinstance(builder_loss, Loss)


def test_metrics_setter_getter():
    # when
    metrics = [Accuracy()]
    builder = ModelBuilder()
    # when
    builder.metrics = metrics
    builder_metrics = builder.metrics
    # then
    assert metrics == builder_metrics
    assert isinstance(builder_metrics, list)


def test_builder_func_setter_getter():
    # given
    def _builder(dimension: List[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = layers.Input(shape=dimension, name="Start")
        out_ = layers.Dense(10)(in_)
        return in_, out_

    builder = ModelBuilder()
    # when
    builder.builder = _builder
    builder_builder = builder.builder
    # then
    assert _builder == builder_builder
    assert isinstance(builder_builder, FunctionType)


def test_sequential_modal_setter_getter():
    # given
    model = Sequential([layers.Dense(10, name="Dense"), layers.Dense(10)])
    builder = ModelBuilder()
    # when
    builder.model = model
    builder_model = builder.model
    # then
    assert model == builder_model
    assert isinstance(builder_model, Model)


def test_metadata_setter_getter():
    # given
    metadata = {"dummy": 2}
    builder = ModelBuilder()
    # when
    builder.metadata = metadata
    builder_metadata = builder.metadata
    # then
    assert metadata == builder_metadata
    assert isinstance(builder_metadata, dict)


def test_wrong_base_dimension_build():
    # given
    dims = "(20, )"
    # when
    try:
        ModelBuilder(dimension=dims)
    # then
    except ValueError:
        return
    assert False


def test_wrong_base_optimizer_build():
    # given
    def _optimizer():
        pass

    # when
    try:
        ModelBuilder(optimizer=_optimizer)
    # then
    except ValueError:
        return
    assert False


def test_wrong_base_loss_build():
    # given
    loss = "def _loss(): pass"
    # when
    try:
        ModelBuilder(loss=loss)
    # then
    except ValueError:
        return
    assert False


def test_wrong_base_metrics_build():
    # given
    metrics = "Accuracy()"
    # when
    try:
        ModelBuilder(metrics=metrics)
    # then
    except ValueError:
        return
    assert False


def test_wrong_base_metrics_list_build():
    # given
    metrics = ["Accuracy()"]
    # when
    try:
        ModelBuilder(metrics=metrics)
    # then
    except ValueError:
        return
    assert False


def test_wrong_base_builder_func_build():
    # given
    builder = "def _builder(): pass"
    # when
    try:
        ModelBuilder(builder=builder)
    # then
    except ValueError:
        return
    assert False


def test_wrong_base_model_build():
    # given
    model = layers.Input(shape=(10,), name="Start")
    # when
    try:
        ModelBuilder(model=model)
    # then
    except ValueError:
        return
    assert False
