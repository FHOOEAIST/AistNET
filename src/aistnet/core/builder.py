"""Base core class of AistNET for setting up a NeuralNetwork training"""  # pylint: disable=R1729

import abc
from types import FunctionType, MethodType
from typing import Any, Dict, List, Tuple, Union

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.models import Functional  # pylint: disable=E0611


class ModelBuilder(abc.ABC):
    """
    ModelBuilder chains all the required options for training
    a TensorFlow|Keras model together with as much flexibility as possible.

    :param kwargs: various model required options, see args

    Args:
        * *dimension* (``tuple``) --
          input dimension for the model without the the batch size
        * *optimizer* (``keras.optimizers.Optimizer | str``) --
          optimizer operator for the training
        * *loss* (``keras.loss.Loss | FunctionType``) --
          loss operator for the error calculation
        * *metrics* (``List[keras.metric.Metric] | List[FunctionType]``) --
          metric operators for additional error calculation without
          inferring with the optimization
        * *builder* (``FunctionType``) --
          a builder method that provides an input and output layer for
          chaining and compiling the model
        * *model* (``keras.model.Sequential``) --
          a already chained mut not compiled model
        * *metadata* (``Dict[Any, Any]``) --
          metadata property for storing additional information to the training

    Examples:

    >>> # Version 1: by implementing a ModelBuilder
    >>> class MyModelBuilder(ModelBuilder):
    >>>     def __init__(self, **kwargs):
    >>>         super().__init__(**kwargs)
    >>>     def build(self) -> (Tuple[layers.Layer], Tuple[layers.Layer]):
    >>>         in_ = layers.Input(10)
    >>>         out_ = layers.Dense(10)(in_)
    >>>         return [in_], [out_]
    >>> builder = MyModelBuilder()
    >>> builder.finalize()

    >>> # Version 2: by using builder function
    >>> def my_builder() -> (Tuple[layers.Layer], Tuple[layers.Layer]):
    >>>     in_ = Input(10)
    >>>     out_ = Dense(10)(in_)
    >>>     return [in_], [out_]
    >>> builder = ModelBuilder(builder=my_builder)
    >>> builder.finalize()

    >>> # Version 3: by using an Sequential model
    >>> my_model = Sequential([
    >>>                        layers.Dense(2, activation="relu", name="layer1"),
    >>>                        layers.Dense(3, activation="relu", name="layer2")
    >>>                        ])
    >>> builder = ModelBuilder(model=my_model)
    >>> builder.finalize()

    """

    _finalized = False
    _dimension = None
    _optimizer = None
    _loss = None
    _metrics = None
    _builder = None
    _model = None
    _metadata: Dict[Any, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.__check_kwarg_and_store("dimension", [tuple, list], kwargs)
        self.__check_kwarg_and_store("optimizer", [Optimizer, str], kwargs, "adam")
        self.__check_kwarg_and_store(
            "loss", [Loss, FunctionType, str], kwargs, "binary_crossentropy"
        )
        self.__check_kwarg_and_store("metrics", [Metric, FunctionType, str], kwargs, [])
        self.__check_kwarg_and_store("builder", [FunctionType], kwargs)
        self.__check_kwarg_and_store("model", [Sequential, Functional], kwargs)
        self.__check_kwarg_and_store("metadata", [dict], kwargs)
        self.__check_kwarg_and_store("finalized", [bool], kwargs)

    def build(  # pylint: disable=R0201
        self,
        dimension: Union[
            List[int],
            Tuple[int],
            List[List[int]],
            Tuple[Tuple[int]],
            List[Tuple[int]],
            Tuple[List[int]],
        ],  # pylint: disable=W0613
    ) -> Tuple[layers.Layer, layers.Layer]:  # pylint: disable=W0613
        """
        Builder method for creating a model that is chained and compiled
        in the finalize step

        :param dimension: (List[int]) containing the input dimension
            without the batch size
        :return: returns the input and output layer of the layer chain
        """
        return None, None

    def finalize(self) -> Model:
        """
        Chain all provided information and compile the model for training

        :return: compiled TensorFlow model
        """
        if self._finalized:
            return self.model
        # check if at least the input dimension is set
        if self.dimension is None:
            raise ValueError("Model building not possible no dimension information")
        # build the model via the method builder
        in_, out_ = self.build(self.dimension)
        # if the method builder is not use check the function builder
        if in_ is None and out_ is None:
            if self.builder is not None and isinstance(self.builder, FunctionType):
                in_, out_ = self.builder(self.dimension)  # pylint: disable=E1102
            # if the function builder is further not use check if a
            # model was provided
            elif self.model is not None:
                self.model.build((None, *self.dimension))
                in_ = self.model.layers[0]
                out_ = self.model.layers[-1]
            else:
                raise ValueError(
                    "No model constructable, none of the builder applicable"
                )
        # run properties check
        self.__check_properties(in_, out_)
        # build final model if it's not a Sequential model
        if not self.model:
            self.model = Model(inputs=[in_], outputs=[out_])
        # compile the model including the optimizer and the other properties
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )
        # show the model result
        self.model.summary()
        return self.model

    @property
    def dimension(
        self,
    ) -> Union[
        List[int],
        Tuple[int],
        List[List[int]],
        Tuple[Tuple[int]],
        List[Tuple[int]],
        Tuple[List[int]],
    ]:
        """
        Input dimension getter

        :return: (``List[int]|Tuple``) the input dimension values
        """
        return self._dimension  # type: ignore

    @dimension.setter
    def dimension(self, dims: List[int]) -> None:
        """Input dimension setter

        :param dims: (``tuple``) containing the input dimension for the final
            model without the batch size
        """
        self._dimension = dims

    @property
    def optimizer(self) -> Union[Optimizer, str]:
        """
        Optimizer getter

        :return: (``Optimizer``) the model to use for training
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """
        Optimizer setter

        :param optimizer: (``keras.optimizers.Optimizer | str``) to use for
            changing the weight in the model
        """
        self._optimizer = optimizer

    @property
    def loss(self) -> Union[Loss, FunctionType, str]:
        """
        Loss getter

        :return: (``Loss``) to calculate the error for the training
        """
        return self._loss

    @loss.setter
    def loss(self, loss: Loss) -> None:
        """Loss setter

        :param loss: (``keras.loss.Loss | FunctionType``) to calculate the
            error to perform the optimization
        """
        self._loss = loss

    @property
    def metrics(self) -> Union[List[Metric], List[str], None]:
        """
        Metrics getter

        :return: (``List[Metric]``) to calculate additionally to the loss error
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[Metric]) -> None:
        """
        Metrics setter

        :param metrics: (``List[keras.metric.Metric] | List[FunctionType]``)
            to calculate additionally to the loss value
        """
        self._metrics = metrics

    @property
    def builder(self) -> MethodType:
        """
        Builder getter

        :return: (``FunctionType``) the function to chain the layer structure
        """
        return self._builder  # type: ignore

    @builder.setter
    def builder(self, builder: FunctionType) -> None:
        """
        Builder setter

        :param builder: (``FunctionType``)
            function to call for building the layer structure
        """
        self._builder = builder

    @property
    def model(self) -> Union[Sequential, Model]:
        """
        Model getter

        :return: (``Sequential``) the sequential model that should be trained
        """
        return self._model

    @model.setter
    def model(self, model: Union[Sequential, Model]) -> None:
        """
        Model setter

        :param model: (``Sequential``)
            sequential model to for compiling to be trainable
        """
        self._model = model

    @property
    def metadata(self) -> Dict[Any, Any]:
        """
        Metadata getter

        :return: (``Dict``) Dictionary with metadata information to the model and the training
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Dict[Any, Any]) -> None:
        """
        Metadata setter

        :param metadata: (``Dict``) Dictionary with metadata information to the model and the training
        """
        self._metadata = metadata

    def __check_kwarg_and_store(
        self, key: str, expected: List[type], kwargs: Any, default: Any = None
    ) -> None:
        """
        Check if a property is provided as named argument and that it is
        valid and store if fine

        :param key: (str) property to check
        :param expected: (List[type]) type of the property to check
        :param kwargs: (dict) named argument dictionary to check if a property
            is insight

        :raises ValueError: this error is thrown if the provided key does not
            match the type
        """
        if key in kwargs.keys():
            self.__check_provided_properties(expected, kwargs[key])
            self.__set_value(key, kwargs[key])
        elif default is not None:
            self.__set_value(key, default)

    def __set_value(self, key: str, value: Any) -> None:
        """
        Set value as property in the current instance

        :param key: name of property
        :param value: value for the property
        """
        self.__check_keras_impl_availability(key, value)
        self.__dict__[f"_{key}"] = value

    @staticmethod
    def __check_keras_impl_availability(key: str, value: Union[Any, List[Any]]) -> None:
        """
        Function for checking given tensorflow type as string as available

        :param key: name of the tensorflow element type
        :param value: list of elements to check
        """
        if key == "optimizer":
            keras_test = tensorflow.keras.optimizers.get
        elif key == "loss":
            keras_test = tensorflow.keras.losses.get
        elif key == "metrics":
            keras_test = tensorflow.keras.metrics.get
        else:
            return

        if isinstance(value, str):
            keras_test(value)
        if isinstance(value, list):
            for v in value:
                if isinstance(v, str):
                    keras_test(v)

    @staticmethod
    def __check_provided_properties(expected: List[type], element: Any) -> None:
        """
        Check if the property is an instance of the expected type

        :param expected: (List[type]) allowed types for the property to check
        :param element: (Any) the element to be check if valid
        """
        if not isinstance(element, list):
            if not any(map(lambda e: isinstance(element, e), expected)):
                raise ValueError(
                    f"Provided type: {type(element)} does not match "
                    f"expected type: {expected}"
                )
        else:
            if not all(
                [any([isinstance(el, ex) for ex in expected]) for el in element]
            ):
                raise ValueError(
                    f"Provided list type: {type(element)} does not match "
                    f"expected list type: {expected}"
                )

    def __check_properties(self, in_: layers.Layer, out_: layers.Layer) -> None:
        """
        Check if the minimal set of properties is valid an can be used for
        model building an compilation

        :param in_: (layers.Layer) input layer of the model
        :param out_: (layers.Layer) output layer of the model

        :raises ValueError: this error is thrown if one of the required
            properties is not satisfied
        """
        if in_ is None:
            raise ValueError(
                "Model build not implemented or doesn't provide a input layer"
            )
        if out_ is None:
            raise ValueError(
                "Model build not implemented or doesn't provide a output layer"
            )
        # check if the optimizer is available and can be read
        if self.optimizer is None:
            raise ValueError(
                "Optimizer is not set but is required for building the model. "
                "`optimizer(str|tensorflow.keras.optimizers)`"
            )
        # check if the provided optimizer as string is valid
        self.__check_keras_impl_availability("optimizer", self.optimizer)
        # check if the loss is available and can be read
        if self.loss is None:
            raise ValueError(
                "Loss method is not set but is required for building the model. "
                "`loss(str|tensorflow.keras.losses.Loss)`"
            )
        # check if the provided loss as string is valid
        self.__check_keras_impl_availability("loss", self.loss)
        # check if metrics are available and can be read
        if self.metrics is None:
            raise ValueError(
                "Metrics are not set and are not required but wrong overridden."
            )
        # check if the provided metrics are strings and valid
        self.__check_keras_impl_availability("metrics", self.metrics)
