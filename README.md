[![GitHub release](https://img.shields.io/github/v/release/fhooeaist/AistNET.svg)](https://github.com/fhooeaist/AistNET/releases)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![PyPI version](https://badge.fury.io/py/aistnet.svg)](https://badge.fury.io/py/aistnet)
[![DOI](https://zenodo.org/badge/389547865.svg)](https://zenodo.org/badge/latestdoi/389547865)

![AistNet](documentation/aist_net.svg)

# AistNET

AistNET (Aist neural network) is a framework for simplifying the creation and training of neural networks
using [Python 3.8](https://www.python.org/)
and [Tensorflow (V. 2.5.0)](https://github.com/tensorflow/tensorflow/tree/v2.5.0) with maximum flexibility. In context
of this tasks AistNET provides interfaces for creating, training and managing trainings of model and to abstracts
buildings block for reusability.

## Current known issues

* **Loading a model is currently only supported on Linux.**

## Main features which are missing in TensorFlow but are available in AistNet

* Create Model the way you like it: Function, Class Method, Sequential
* Add custom functions and use them without any issue (except callbacks because
  of [tensorflow:36635](https://github.com/tensorflow/tensorflow/pull/36635) issue)
* Automatic saving of all relevant information
  * Model: H5 and ProtoBuf version and as JSON definition
  * System: from TensorFlow version to used callbacks
  * Custom functions are saved along with the model and the system
* Restore where you stopped and resume the training including custom functions

## Getting Started

To install the current release use pip:

```pip
pip install aistnet
```

To update AistNET to the latest version, add `--upgrade` flag to the above command.

To create your first model such as a Dense-Net or U-NET using AistNET follow the examples:

### With Sequential Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from aistnet.core.builder import ModelBuilder

linear = Sequential([
  layers.Dense(2, activation="relu", name="layer1"),
  layers.Dense(3, activation="relu", name="layer2")
])
dims = (28,)
optimizer = Adam()
loss = BinaryCrossentropy()
builder = ModelBuilder(dimension=dims, model=linear,
                       optimizer=optimizer, loss=loss)
model = builder.finalize()
```

### With Builder Function

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from aistnet.core.builder import ModelBuilder
from aistnet.architectures.unet import cnn_2d_auto_encoder_with_skip

dims = [240, 224, 1]
builder_function = cnn_2d_auto_encoder_with_skip(blocks=2)
optimizer = Adam()
loss = BinaryCrossentropy()
builder = ModelBuilder(dimension=dims, builder=builder_function,
                       optimizer=optimizer, loss=loss)
model = builder.finalize()
```

The model can now be trained normally via the TensorFlow api.

### Train your model with the Trainer and use the automatic tracing and saving capabilities:

After creation, you can train the model as usually:

```python
import tempfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from aistnet.core.builder import ModelBuilder
from aistnet.core.trainer import Trainer

linear = Sequential([
  layers.Dense(2, activation="relu", name="layer1"),
  layers.Dense(3, activation="relu", name="layer2")
])
dims = (28,)
optimizer = Adam()
loss = BinaryCrossentropy()
builder = ModelBuilder(dimension=dims, model=linear,
                       optimizer=optimizer, loss=loss)
trainer = Trainer(builder=builder, store_path=tempfile.TemporaryDirectory().name)
trainer.fit(
  x=tf.convert_to_tensor([1, 2, 3, 4, 5]),
  y=tf.convert_to_tensor([2, 3, 4, 5, 6]),
  batch_size=16,
  epochs=10,
  validation_data=(
    tf.convert_to_tensor([1, 2, 3, 4, 5]),
    tf.convert_to_tensor([2, 3, 4, 5, 6]),
  ),
)
```

This runs the training of the model but also saves metric information, and the model itself to the file system.

Finally, the model can be used or restored like this:

```python
import tempfile
import tensorflow as tf
from aistnet.core.trainer import Trainer

builder, trainer = Trainer.load(tempfile.TemporaryDirectory().name)
trainer.fit(
  x=tf.convert_to_tensor([1, 2, 3, 4, 5]),
  y=tf.convert_to_tensor([2, 3, 4, 5, 6]),
  batch_size=16,
  epochs=20,
  initial_epoch=10,
  validation_data=(
    tf.convert_to_tensor([1, 2, 3, 4, 5]),
    tf.convert_to_tensor([2, 3, 4, 5, 6]),
  ),
)
```

### Use your own loss function

AistNET lets you create or own loss function and other custom implementations. It tries to automatically locate them and
to save them along with the model and the configuration. Further it restores the custom implementations with the loading
of a saved state.

```python
from typing import Tuple
import tempfile
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense

from aistnet.core.builder import ModelBuilder
from aistnet.core.trainer import Trainer

store_path = tempfile.TemporaryDirectory().name
dims = (1,)
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


builder = ModelBuilder(
  builder=build, dimension=dims, optimizer=optimizer, loss=loss, metrics=metrics
)
trainer = Trainer(builder=builder, store_path=store_path)
# train and save the state
trainer.fit(
  x=tf.convert_to_tensor([1, 2, 3, 4, 5]),
  y=tf.convert_to_tensor([2, 3, 4, 5, 6]),
  batch_size=16,
  epochs=10,
  validation_data=(
    tf.convert_to_tensor([1, 2, 3, 4, 5]),
    tf.convert_to_tensor([2, 3, 4, 5, 6]),
  ),
)

# load the previous state and continue training in a new session
builder_new, trainer_new = Trainer.load(store_path)
x_true = tf.convert_to_tensor([[1.0]])
x_pred = tf.convert_to_tensor([[1.0]])
# check the reconstucted custom loss function and the previous epoch state
print(builder_new.loss(x_true, x_pred) == loss(x_true, x_pred))
print(trainer_new.run_metadata["epochs"] == 10)

builder_new.model.fit(
  x=tf.convert_to_tensor([1, 2, 3, 4, 5]),
  y=tf.convert_to_tensor([2, 3, 4, 5, 6]),
  batch_size=16,
  epochs=20,
  initial_epoch=10,
  validation_data=(
    tf.convert_to_tensor([1, 2, 3, 4, 5]),
    tf.convert_to_tensor([2, 3, 4, 5, 6]),
  ),
)
```

## FAQ

1. Why another Tensorflow wrapper?

The reason for AistNET is the simplification of neural networks. It provides functionality to build, parameterize and
train models with any architecture. The model can be customized in every way.

2. Is there a possibility to use AistNET with other frameworks like [PyTorch](https://pytorch.org/)?

No currently, AistNET only supports Tensorflow. If you want to use PyTorch we
recommend [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), which follows a similar wrapping
philosophy.

3. Does AistNET support any other model architectures?

For the moment AistNET has a builder function for U-Nets with skip layers. But we are going to extend AistNET step by
step.

## Contributing

**First make sure to read our general [contribution guidelines](https://fhooeaist.github.io/CONTRIBUTING.html).**

In addition to that, the following applies to this repository:

- Due to the focus on performance dependencies (except as bridges to other loggers) are not allowed. IF you have a very
  good reason to add a dependency please state so in the corresponding issue / pull request.

## Licence

Copyright (c) 2020 the original author or authors. DO NOT ALTER OR REMOVE COPYRIGHT NOTICES.

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

## Research

If you are going to use this project as part of a research paper, we would ask you to reference this project by citing
it.

[![DOI](https://zenodo.org/badge/389547865.svg)](https://zenodo.org/badge/latestdoi/389547865)
