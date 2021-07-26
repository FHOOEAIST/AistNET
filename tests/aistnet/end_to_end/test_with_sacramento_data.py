# pylint: disable=W0236,R0801
import pathlib
import shutil
from typing import List, Tuple, Union

import pandas as pd
from tensorflow.keras import Input, Sequential, layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss, MeanSquaredError
from tensorflow.keras.optimizers import Adam, Optimizer

from aistnet.core.builder import ModelBuilder
from aistnet.core.trainer import Trainer


# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
def test_end_to_end_with_sacramento_data_by_array_and_sequence():
    # given
    data = pd.read_csv(
        f"{pathlib.Path(__file__).parent.absolute()}/sacramentorealestatetransactions.csv",
        sep=",",
    )
    data = data[["beds", "baths", "sq__ft", "price"]]
    data_train = data.sample(frac=0.7, random_state=42)
    data_validate = data.drop(data_train.index).to_numpy()
    data_train = data_train.to_numpy()

    model = Sequential()
    model.add(Dense(12, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))

    dims = (3,)
    optimizer = Adam()
    loss = MeanSquaredError()

    builder = ModelBuilder(model=model)
    builder.loss = loss

    builder = ModelBuilder(dimension=dims, model=model, optimizer=optimizer, loss=loss)
    trainer = Trainer(builder=builder)
    trainer.fit(
        x=data_train[:, :3],
        y=data_train[:, 3],
        batch_size=16,
        epochs=10,
        validation_data=(data_validate[:, :3], data_validate[:, 3]),
    )


def test_end_to_end_with_sacramento_data_by_array_by_builder():
    # given
    data = pd.read_csv(
        f"{pathlib.Path(__file__).parent.absolute()}/sacramentorealestatetransactions.csv",
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

    store_path = f"{pathlib.Path(__file__).parent.absolute()}/sacramento_end_to_end"
    shutil.rmtree(store_path, ignore_errors=True)

    builder = MyModelBuilder()
    trainer = Trainer(builder=builder, store_path=store_path)
    trainer.fit(
        x=data_train[:, :3],
        y=data_train[:, 3],
        batch_size=16,
        epochs=10,
        validation_data=(data_validate[:, :3], data_validate[:, 3]),
    )
