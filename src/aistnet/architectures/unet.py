"""Main module for testing"""

from typing import Callable, List, Tuple

from tensorflow.keras import layers

import aistnet.layers.layers as grouped_blocks


def cnn_2d_auto_encoder_with_skip(  # pylint: disable=R0913
    blocks: int,
    initial_kernel: int = 16,
    normalize_encoder: bool = True,
    normalize_decoder: bool = True,
    normalize_bottom: bool = True,
    drop_factor: float = 0.2,
    bottom_drop_factor: float = 0.5,
    final_activation: str = "relu",
) -> Callable[[List[int]], Tuple[layers.Layer, layers.Layer]]:
    """
    Dynamic model builder for a given number of blocks for the networks depth

    :param normalize_encoder: enable batch normalization in encoder blocks
    :param normalize_decoder: enable batch normalization in decoder blocks
    :param normalize_bottom: enable batch normalization in bottom block
    :param final_activation: final activation function to use for the output
    :param bottom_drop_factor: factor for the bottom dropout layer in the
        u-net, must be in range [0, 1[
    :param drop_factor: factor for the dropout layer, must be in range [0, 1[
    :param initial_kernel: initial number of kernels in the first CNN layer
    :param blocks: number of encoding and decoding blocks to build
    :return: Callable function for creating the actual network
    """

    if initial_kernel % 2 != 0:
        raise ValueError(
            "Initial kernel value is not valid, only even "
            "numbers are allowed: 4, 6, 8, …"
        )

    if drop_factor >= 1:
        raise ValueError("Dropout factor is not in range, only range [0, 1[ is allowed")

    if bottom_drop_factor >= 1:
        raise ValueError(
            "Bottom dropout factor is not in range, only range [0, 1[ is allowed"
        )

    def _creator(dimensions: List[int]) -> Tuple[layers.Layer, layers.Layer]:
        """
        Network builder for the outer given number of sampling blocks

        :param dimensions: Model input dimensions
        :return: tuple with input and output layer
        """
        input_, p = grouped_blocks.input_block(dimensions)

        down_block = []
        # reduce the kernels by half because each block increases it by
        # factor 2 initially
        size = initial_kernel // 2
        # create the number of encoders
        for _ in range(blocks):
            p, size, b = grouped_blocks.conv_down_block(
                p, size, normalize_encoder, drop_factor
            )
            down_block.append((p, size, b))
        # create bottom layer
        p, size = grouped_blocks.conv_base_block(
            p, size, normalize_bottom, bottom_drop_factor
        )
        # create the same number of decoders to the encoders to reach the same size
        # each layer reduces the number of kernels by factor of 2
        for i in reversed(range(blocks)):
            skip_block = down_block[i]
            p, size = grouped_blocks.conv_up_block(
                p, skip_block[2], size, normalize_decoder, drop_factor
            )
        # create end block for finishing up the auto encoder
        output = grouped_blocks.end_block(p, final_activation)
        return input_, output

    return _creator
