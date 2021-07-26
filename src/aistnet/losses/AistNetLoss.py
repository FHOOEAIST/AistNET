"""AistNetLoss Wrapper for Loss Classes"""  # pylint: disable=C0103,E0611
from typing import Any, Callable, Dict

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils.losses_utils import ReductionV2


class AistNetLoss(LossFunctionWrapper):
    """
    AistNetLoss Wrapper to abstract the reduction parameter
    """

    def __init__(
        self,
        fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        name: str,
        reduction: ReductionV2 = losses_utils.ReductionV2.AUTO,
        **kwargs: Dict[str, Any]
    ):
        """
        AistNetLoss is a thin wrapper over LossFunctionWrapper to
        set ReductionV2.AUTO per default

        :param fn: callable loss function
        :param name: textual name of the loss function
        :param reduction: type of the loss reduction type
        :param kwargs: various additional options, see LossFunctionWrapper
        """
        super().__init__(fn, reduction, name, **kwargs)
