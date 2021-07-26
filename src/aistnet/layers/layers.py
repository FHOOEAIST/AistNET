"""Reusable Layer definitions"""

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dropout,
    Input,
    Lambda,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)


def conv_down_block(
    s: layers.Layer,
    pre_kernel_size: int,
    normalize: bool = True,
    drop_factor: float = 0.2,
) -> Tuple[layers.Layer, int, layers.Layer]:
    """
    Convolution down sampling layer for stretching the image content

    :param drop_factor: factor for the dropout layer, must be in range ]0, 1[
    :param normalize: enable batch normalization in all grouped blocks
    :param s: previous layer as input to the current block
    :param pre_kernel_size: number of kernels used in the previous layer
    :return: tuple with the resulting last conv layer for skipping to the up
        sampling, the number of used kernels and the final layer
    """
    if drop_factor <= 0 or drop_factor >= 1:
        raise ValueError("Dropout factor is not in range, only range ]0, 1[ is allowed")

    current_kernel_size = pre_kernel_size * 2
    c1 = Conv2D(
        current_kernel_size,
        3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(s)
    if normalize:
        c1 = BatchNormalization()(c1)
    c1 = Conv2D(
        current_kernel_size,
        3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c1)
    if normalize:
        c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    c1 = Dropout(drop_factor)(c1)
    return p1, current_kernel_size, c1


def conv_base_block(
    s: layers.Layer,
    pre_kernel_size: int,
    normalize: bool = True,
    bottom_drop_factor: float = 0.5,
) -> Tuple[layers.Layer, int]:
    """
    Unet convolution base layer

    :param bottom_drop_factor: factor for the bottom dropout layer in the u-net,
        must be in range ]0, 1[
    :param normalize: enable batch normalization in all grouped blocks
    :param s: previous layer as input to the current block
    :param pre_kernel_size: number of kernels used in the previous layer
    :return: tuple with the final layer and the number of used kernels
    """
    if bottom_drop_factor <= 0 or bottom_drop_factor >= 1:
        raise ValueError(
            "Bottom dropout factor is not in range, only range ]0, 1[ is allowed"
        )

    current_kernel_size = pre_kernel_size * 2
    c5 = Conv2D(
        current_kernel_size,
        3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(s)
    if normalize:
        c5 = BatchNormalization()(c5)
    c5 = Conv2D(
        current_kernel_size,
        3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c5)
    if normalize:
        c5 = BatchNormalization()(c5)
    c5 = Dropout(bottom_drop_factor)(c5)
    return c5, current_kernel_size


def conv_up_block(
    s: layers.Layer,
    p: layers.Layer,
    pre_kernel_size: int,
    normalize: bool = True,
    drop_factor: float = 0.2,
) -> Tuple[layers.Layer, int]:
    """
    Convolution up sampling layer for rebuilding the image content

    :param drop_factor: factor for the dropout layer, must be in range ]0, 1[
    :param normalize: enable batch normalization in all grouped blocks
    :param s: previous layer as input to the current block
    :param p: the skipping layer for propagating the original values
    :param pre_kernel_size: number of kernels used in the previous layer
    :return: tuple with the final layer and the used kernels
    """
    if drop_factor <= 0 or drop_factor >= 1:
        raise ValueError("Dropout factor is not in range, only range ]0, 1[ is allowed")

    second_kernel_size = pre_kernel_size // 2
    u6 = UpSampling2D(size=(2, 2))(s)
    u6 = Conv2D(
        second_kernel_size,
        2,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u6)
    u6 = concatenate([p, u6])
    c6 = Conv2D(
        second_kernel_size,
        3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u6)
    if normalize:
        c6 = BatchNormalization()(c6)
    c6 = Dropout(drop_factor)(c6)
    c6 = Conv2D(
        second_kernel_size,
        3,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c6)
    if normalize:
        c6 = BatchNormalization()(c6)

    return c6, second_kernel_size


def input_block(dimensions: List[int]) -> Tuple[layers.Layer, layers.Layer]:
    """
    Input block for any CNN with automatic conversion from image to float and
    range [0, 255] to [0., 1.]

    :param dimensions: tuple defining the model input dimensions
    :return: tuple with the input layer for chaining the final model and the
        final layer of the block
    """

    inputs = Input(dimensions, dtype=tf.int8)
    s = Lambda(lambda x: x / 255)(inputs)
    return inputs, s


def end_block(s: layers.Layer, activation: str = "relu") -> layers.Layer:
    """
    Final block for a u-net to resample the original image dimensions

    :param activation: final activation function to use for the output
    :param s: previous layer as input to the current block
    :return: the final layer of this block
    """
    conv = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(s)
    return Conv2D(1, 1, activation=activation)(conv)
