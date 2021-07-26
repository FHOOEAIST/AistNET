# pylint: disable=W0236,R0801
import pathlib
import shutil
from typing import List, Tuple, Union

import dill
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer

from aistnet.core.builder import ModelBuilder
from aistnet.core.trainer import Trainer


def test_end_to_end_with_custom_callback():
    # given
    data = pd.read_csv(
        f"{pathlib.Path(__file__).parent.absolute()}/../end_to_end/sacramentorealestatetransactions.csv",
        sep=",",
    )
    data = data[["beds", "baths", "sq__ft", "price"]]
    data_train = data.sample(frac=0.7, random_state=42)
    data_validate = data.drop(data_train.index).to_numpy()
    data_train = data_train.to_numpy()

    dims = (3,)

    class MyModelBuilder(ModelBuilder):
        def build(self, dimension: Tuple[int]) -> Tuple[layers.Layer, layers.Layer]:
            in_ = Input(shape=dimension)
            d1 = Dense(12, activation="relu")(in_)
            d2 = Dense(8, activation="relu")(d1)
            d3 = Dense(1)(d2)
            return in_, d3

        @ModelBuilder.dimension.getter
        def dimension(
            self,
        ) -> Union[List[int], Tuple[int], List[List[int]], Tuple[Tuple[int]]]:
            return dims

        @ModelBuilder.optimizer.getter
        def optimizer(self) -> Optimizer:
            return "adam"

        @ModelBuilder.loss.getter
        def loss(self) -> Loss:
            return "mean_squared_error"

    log_path = f"{pathlib.Path(__file__).parent.absolute()}/sacramento_end_to_end"
    shutil.rmtree(log_path, ignore_errors=True)

    builder = MyModelBuilder()
    trainer = Trainer(builder=builder, store_path=log_path)
    trainer.fit(
        x=data_train[:, :3],
        y=data_train[:, 3],
        batch_size=16,
        epochs=10,
        validation_data=(data_validate[:, :3], data_validate[:, 3]),
        callbacks=[TensorBoard(log_dir=log_path)],
    )


def test_end_to_end_with_custom_implementation():
    # given
    data = pd.read_csv(
        f"{pathlib.Path(__file__).parent.absolute()}/../end_to_end/sacramentorealestatetransactions.csv",
        sep=",",
    )
    data = data[["beds", "baths", "sq__ft", "price"]]
    data["price"] = data["price"].astype(float)
    data_train = data.sample(frac=0.7, random_state=42)
    data_validate = data.drop(data_train.index).to_numpy()
    data_train = data_train.to_numpy()

    dims = (3,)
    optimizer = "adam"

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return (y_true - y_pred) ** 2

    def build(dimension: Tuple[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = Input(shape=dimension)
        d1 = Dense(12, activation="relu")(in_)
        d2 = Dense(8, activation="relu")(d1)
        d3 = Dense(1)(d2)
        return in_, d3

    log_path = f"{pathlib.Path(__file__).parent.absolute()}/sacramento_end_to_end"
    shutil.rmtree(log_path, ignore_errors=True)

    builder = ModelBuilder(
        builder=build, dimension=dims, optimizer=optimizer, loss=loss
    )
    trainer = Trainer(builder=builder, store_path=log_path)
    trainer.fit(
        x=tf.convert_to_tensor(data_train[:, :3]),
        y=tf.convert_to_tensor(data_train[:, 3]),
        batch_size=16,
        epochs=10,
        validation_data=(
            tf.convert_to_tensor(data_validate[:, :3]),
            tf.convert_to_tensor(data_validate[:, 3]),
        ),
    )

    builder_new, _ = Trainer.load(log_path)
    x_true = tf.convert_to_tensor([[1.0]])
    x_pred = tf.convert_to_tensor([[1.0]])
    assert builder_new.loss(x_true, x_pred) == loss(x_true, x_pred)


def test_end_to_end_with_2_runs():
    # given
    data = pd.read_csv(
        f"{pathlib.Path(__file__).parent.absolute()}/../end_to_end/sacramentorealestatetransactions.csv",
        sep=",",
    )
    data = data[["beds", "baths", "sq__ft", "price"]]
    data["price"] = data["price"].astype(float)
    data_train = data.sample(frac=0.7, random_state=42)
    data_validate = data.drop(data_train.index).to_numpy()
    data_train = data_train.to_numpy()

    dims = (3,)
    optimizer = "adam"
    metrics = ["accuracy"]

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return (y_true - y_pred) ** 2

    def build(dimension: Tuple[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = Input(shape=dimension)
        d1 = Dense(12, activation="relu")(in_)
        d2 = Dense(8, activation="relu")(d1)
        d3 = Dense(1)(d2)
        return in_, d3

    log_path = f"{pathlib.Path(__file__).parent.absolute()}/sacramento_end_to_end"
    shutil.rmtree(log_path, ignore_errors=True)

    builder = ModelBuilder(
        builder=build, dimension=dims, optimizer=optimizer, loss=loss, metrics=metrics
    )
    trainer = Trainer(builder=builder, store_path=log_path)
    trainer.fit(
        x=tf.convert_to_tensor(data_train[:, :3]),
        y=tf.convert_to_tensor(data_train[:, 3]),
        batch_size=16,
        epochs=10,
        validation_data=(
            tf.convert_to_tensor(data_validate[:, :3]),
            tf.convert_to_tensor(data_validate[:, 3]),
        ),
    )

    trainer.fit(
        x=tf.convert_to_tensor(data_train[:, :3]),
        y=tf.convert_to_tensor(data_train[:, 3]),
        batch_size=16,
        epochs=20,
        initial_epoch=10,
        validation_data=(
            tf.convert_to_tensor(data_validate[:, :3]),
            tf.convert_to_tensor(data_validate[:, 3]),
        ),
    )

    builder_new, trainer_new = Trainer.load(log_path)
    x_true = tf.convert_to_tensor([[1.0]])
    x_pred = tf.convert_to_tensor([[1.0]])
    assert builder_new.loss(x_true, x_pred) == loss(x_true, x_pred)
    assert trainer_new.run_metadata["epochs"] == 20

    builder_new.model.fit(
        x=tf.convert_to_tensor(data_train[:, :3]),
        y=tf.convert_to_tensor(data_train[:, 3]),
        batch_size=16,
        epochs=30,
        initial_epoch=20,
        validation_data=(
            tf.convert_to_tensor(data_validate[:, :3]),
            tf.convert_to_tensor(data_validate[:, 3]),
        ),
    )


def test_dill_marshaling():
    data = pd.read_csv(
        f"{pathlib.Path(__file__).parent.absolute()}/../end_to_end/sacramentorealestatetransactions.csv",
        sep=",",
    )
    data = data[["beds", "baths", "sq__ft", "price"]]
    data["price"] = data["price"].astype(float)
    data_train = data.sample(frac=0.7, random_state=42)
    data_validate = data.drop(data_train.index).to_numpy()
    data_train = data_train.to_numpy()

    dims = (3,)
    optimizer = "adam"

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return (y_true - y_pred) ** 2

    def build(dimension: Tuple[int]) -> Tuple[layers.Layer, layers.Layer]:
        in_ = Input(shape=dimension)
        d1 = Dense(12, activation="relu")(in_)
        d2 = Dense(8, activation="relu")(d1)
        d3 = Dense(1)(d2)
        return in_, d3

    in_, out_ = build(dims)
    model = Model(inputs=[in_], outputs=[out_])
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(
        x=tf.convert_to_tensor(data_train[:, :3]),
        y=tf.convert_to_tensor(data_train[:, 3]),
        batch_size=16,
        epochs=10,
        validation_data=(
            tf.convert_to_tensor(data_validate[:, :3]),
            tf.convert_to_tensor(data_validate[:, 3]),
        ),
    )
    model.fit(
        x=tf.convert_to_tensor(data_train[:, :3]),
        y=tf.convert_to_tensor(data_train[:, 3]),
        batch_size=16,
        epochs=20,
        initial_epoch=10,
        validation_data=(
            tf.convert_to_tensor(data_validate[:, :3]),
            tf.convert_to_tensor(data_validate[:, 3]),
        ),
    )
    path = f"{pathlib.Path(__file__).parent.absolute()}/sacramento_end_to_end.h5"
    model.save(path)
    loss2 = dill.loads(dill.dumps(loss))
    model2 = tf.keras.models.load_model(path, custom_objects={"loss": loss2})
    model2.fit(
        x=tf.convert_to_tensor(data_train[:, :3]),
        y=tf.convert_to_tensor(data_train[:, 3]),
        batch_size=16,
        epochs=30,
        initial_epoch=20,
        validation_data=(
            tf.convert_to_tensor(data_validate[:, :3]),
            tf.convert_to_tensor(data_validate[:, 3]),
        ),
    )
