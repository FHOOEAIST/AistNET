# type: ignore[attr-defined]
"""Base core class of AistNET for training NeuralNetwork"""

import abc
import inspect
import json
import tempfile
from pathlib import Path
from types import FunctionType
from typing import Any, Dict, List

import dill  # nosec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback, History, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model

from aistnet.core.builder import ModelBuilder


class Trainer(abc.ABC):
    """
    Trainer wraps the fit operation by tensorflow to simplify
    training and saving

    :param builder: an instance of a ModelBuilder that should be trained
    """

    __log_path = "{}"
    __checkpoint_path = "{}/checkpoints/checkpoint"
    __base_path = "{}/run"
    __config_path = __base_path + "/config"
    __model_json_path = __config_path + "/model_config.json"
    __system_json_path = __config_path + "/system_config.json"
    __protobuf_model_path = __base_path + "/final"
    __h5_model_path = __base_path + "/final.h5"
    __dir_for_custom_element_path = __config_path + "/{}"
    __custom_element_path = __dir_for_custom_element_path + "/{}.func"
    __callback_path = __config_path + "/callback"
    __callback_element_path = __callback_path + "/{}.callback"

    def __init__(self, builder: ModelBuilder, store_path: str = None):
        self._model = None
        self.builder = builder
        self.callbacks: Dict[str, keras.callbacks.Callback] = {}
        self.run_metadata: Dict[str, Any] = {}
        if store_path is None:
            self.base_store_path = tempfile.TemporaryDirectory().name
        else:
            self.base_store_path = store_path
        self.store_path = self.base_store_path

    @property
    def model(self) -> Model:
        """
        Model property getter

        :return: the current model in the trainer
        """
        return self._model

    def add_callbacks(self, callbacks: List[keras.callbacks.Callback]) -> List[str]:
        """
        Add callbacks to apply during training

        Note `tf.keras.callbacks.ProgbarLogger`
            and `tf.keras.callbacks.History` callbacks are created automatically

        :param callbacks: Dict of callbacks with name to apply during training.
        :return: None
        """
        keys = []
        for v in callbacks:
            k = v.__class__.__name__
            self.callbacks[k] = v
            keys.append(k)
        return keys

    def drop_callbacks(self, callbacks: List[str]) -> None:
        """
        Remove callbacks to not apply them during training

        :param callbacks: List of str
        :return: None
        """
        for k in callbacks:
            if k in self.callbacks.keys():
                del self.callbacks[k]

    def fit(
            self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose="auto",
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
    ) -> History:
        """TensorFlow Keras fit function see

        :func:`tensorflow.python.keras.engine.training.Model.fit`

        Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

        """
        # grep training parameters to protocol them
        self._set_exists("batch_size", batch_size)
        self._set_exists("epochs", epochs)
        self._set_exists("initial_epoch", initial_epoch)
        self._set_exists("steps_per_epoch", steps_per_epoch)
        self._set_exists("validation_steps", validation_steps)
        self._set_exists("validation_batch_size", validation_batch_size)
        self._set_exists("shuffle", shuffle)
        self._set_exists("verbose", verbose)
        start = 0
        if "initial_epoch" in self.run_metadata:
            start = self.run_metadata["initial_epoch"]
        end = self.run_metadata["epochs"]
        run_name = f"/epochs-{start}-{end}"
        self.store_path = self.base_store_path + run_name

        # Setup logging and storage path
        Path(self.store_path).mkdir(parents=True, exist_ok=False)
        print(f"Workdir is set to {self.store_path}")
        print(
            f"TensorBoard log dir is set to {Trainer.__log_path.format(self.store_path)}"
        )
        print(
            f"ModelCheckpoint log dir is set to {Trainer.__checkpoint_path.format(self.store_path)}"
        )
        # add TensorBoard logging as default
        self.add_callbacks(
            [
                TensorBoard(Trainer.__log_path.format(self.store_path)),
                ModelCheckpoint(
                    Trainer.__checkpoint_path.format(self.store_path),
                    save_weights_only=True,
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]
        )

        # FUTURE get dimension from data if not available
        self._model = self.builder.finalize()
        # setup callbacks
        if callbacks is not None:
            for callback in callbacks:
                custom_elements, named_elements = self.__save_custom_functions(
                    [callback], "", "", False
                )
                if len(custom_elements) > 0:
                    self.callbacks[custom_elements[0]] = callback
                elif len(named_elements) > 0:
                    self.callbacks[named_elements[0]] = callback
                else:
                    raise ValueError("Not supported callback")

        history = self._model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=list(self.callbacks.values()),
            validation_split=validation_split,
            validation_data=validation_data,
            shuffle=shuffle,
            class_weight=class_weight,
            sample_weight=sample_weight,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_batch_size=validation_batch_size,
            validation_freq=validation_freq,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        self.save()
        return history

    def _set_exists(self, name: str, value: Any):
        """
        Function that sets a given value from a dictionary as property in the trainer
        """
        if value is not None:
            self.run_metadata[name] = value

    def save(self) -> None:
        """
        Save the model the two provided formats by tensorflow
        to the given path.

        :param user_path: name with path to the directory to store the model and configuration
        """
        # save model as json element
        Path(Trainer.__config_path.format(self.store_path)).mkdir(
            parents=True, exist_ok=True
        )
        with open(Trainer.__model_json_path.format(self.store_path), "w") as writer:
            writer.write(self._model.to_json())
        # collect basic tensorflow setup information
        system_config = {
            "tensorflow_version": tf.__version__,
            "tensorflow_git_version": tf.__git_version__,
            "tensorflow_compile_version": tf.__compiler_version__,
            "keras_version": tf.keras.__version__,
            "model_inputs": [i.shape.as_list() for i in self._model.inputs],
            "model_outputs": [i.shape.as_list() for i in self._model.outputs],
            "metadata": self.builder.metadata,
            "run_metadata": self.run_metadata,
            "base_store_path": self.base_store_path,
        }
        # save custom elements to allow a full restore
        for e, n in [
            ([self._model.optimizer], "optimizer"),
            ([self._model.loss], "loss"),
            (self._model.metrics, "metrics"),
            # disabled callback restore
            # https://github.com/tensorflow/tensorflow/pull/36635
            # (self.callbacks.values(), "callbacks", False),
        ]:
            custom_elements, named_elements = Trainer.__save_custom_functions(
                e, self.store_path, n
            )
            system_config[f"custom_{n}"] = custom_elements
            system_config[n] = named_elements
        # save setup information as json
        with open(Trainer.__system_json_path.format(self.store_path), "w") as writer:
            writer.write(json.dumps(system_config))
        # save the model as checkpoint
        self._model.save(
            Trainer.__protobuf_model_path.format(self.store_path), save_format="tf"
        )
        # save the model as keras h5 file
        self._model.save(Trainer.__h5_model_path.format(self.store_path))

    @staticmethod
    def __save_custom_functions(
        elements: List[Any], user_path: str, custom_type_name: str, save: bool = True
    ) -> (List[str], List[str]):
        """
        Save custom elements connected to the model into
        marshaled object for reloading and returns
        two lists with custom and standard elements

        :param elements: (``List[Any]``) containing custom and non custom elements
        :param user_path: (``str``) path as string for storing the custom elements into
        :param custom_type_name: (``str``) type of the custom and non custom elements
        :param save: (``bool``) flag if the storing should be executed or only the check
        """

        custom_elements = []
        named_elements = []
        for e in elements:
            if type(e).__name__ == "str":
                named_elements.append(e)
            elif type(e).__module__.count("tensorflow") > 0:
                name = type(e).__name__
                if "Wrapper" not in name:
                    named_elements.append(name)
            elif type(e).__name__ == "function":
                custom_element_name = e.__name__
                return_type = inspect.signature(e).return_annotation
                if return_type is None:
                    return_type = "<class 'tensorflow.python.framework.ops.Tensor'>"
                custom_elements.append((custom_element_name, str(return_type)))
                if save:
                    Trainer.__save_instance(
                        e, user_path, custom_type_name, custom_element_name
                    )
            elif isinstance(e, Callback):
                custom_element_name = e.__class__.__name__
                custom_elements.append(
                    (custom_element_name, f"<class '{e.__class__}'>")
                )
                if save:
                    Trainer.__save_instance(
                        e, user_path, custom_type_name, custom_element_name
                    )

        return custom_elements, named_elements

    @staticmethod
    def __save_instance(
            element: Any, user_path: str, custom_type_name: str, custom_element_name: str
    ):
        Path(
            Trainer.__dir_for_custom_element_path.format(user_path, custom_type_name)
        ).mkdir(parents=True, exist_ok=True)
        with open(
                Trainer.__custom_element_path.format(
                    user_path, custom_type_name, custom_element_name
                ),
                "wb",
        ) as writer:
            dill.dump(element, writer)

    @staticmethod
    def load(user_path: str):
        """
        Load method for restoring an training state with ModelBuilder and Trainer

        :param user_path: user path to the saved model structure
        """
        if not Path(Trainer.__system_json_path.format(user_path)).exists():
            print(
                "No unique run to restore - falling back to latest run by creation date"
            )
            sorted_runs = [
                (str(p), p.stat().st_ctime) for p in Path(user_path).glob("*")
            ]
            sorted_runs.sort(key=lambda x: x[1], reverse=True)
            user_path = sorted_runs[0][0]
            print(f"Restoring: {user_path}")

        with open(Trainer.__system_json_path.format(user_path), "r") as reader:
            system_json_def = dict(json.load(reader))

        # restore input dimension
        model_inputs = [tuple(i) for i in system_json_def["model_inputs"]]
        if len(model_inputs) == 1:
            model_inputs = model_inputs[0]

        # restored custom functions
        custom_element_with_function = {}
        # recover all custom functions
        for k, v in system_json_def.items():
            if "custom_" in k:
                custom_type_name = k.replace("custom_", "")
                if len(v) > 0:
                    custom_element_with_function[custom_type_name] = {}
                for custom_element_name, _ in v:
                    code = Trainer.__load_custom_functions(
                        user_path, custom_type_name, custom_element_name
                    )
                    custom_element_with_function[custom_type_name][
                        custom_element_name
                    ] = code

        # restore optimizer
        optimizer = Trainer.__get_custom_or_named(
            "optimizer", True, custom_element_with_function, system_json_def
        )

        # restore loss
        loss = Trainer.__get_custom_or_named(
            "loss", True, custom_element_with_function, system_json_def
        )

        # restore metrics
        metrics = Trainer.__get_custom_or_named(
            "metrics", False, custom_element_with_function, system_json_def
        )

        # restore metadata
        metadata = system_json_def["metadata"]
        run_metadata = system_json_def["run_metadata"]
        base_store_path = system_json_def["base_store_path"]

        # disabled callback restore
        # https://github.com/tensorflow/tensorflow/pull/36635
        # callbacks = [
        #     c
        #     for c in Trainer.__get_custom_or_named(
        #         "callbacks", False, custom_element_with_function, system_json_def
        #     )
        #     if not isinstance(c, str)
        # ]

        saved_custom_objects = {}
        for _, custom_type in custom_element_with_function.items():
            for key, value in custom_type.items():
                saved_custom_objects[key] = value

        # load model with all custom functions
        final_model = keras.models.load_model(
            Trainer.__h5_model_path.format(user_path),
            custom_objects=saved_custom_objects,
        )

        # restore ModelBuilder
        builder = ModelBuilder(
            model=final_model,
            finalized=True,
            dimension=model_inputs,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            metadata=metadata,
        )

        # restore Trainer
        trainer = Trainer(builder=builder, store_path=base_store_path)
        # disabled callback restore
        # https://github.com/tensorflow/tensorflow/pull/36635
        # trainer.add_callbacks(callbacks)
        trainer.run_metadata = run_metadata

        return builder, trainer

    @staticmethod
    def __get_custom_or_named(
            key: str,
            first_only: bool,
            custom_functions: Dict[str, Dict[str, FunctionType]],
            system_def: Dict[str, List[str]],
    ) -> Any:
        """
        Restore custom and named functions from system def and the restored custom functions

        :param key: type to restore such as dimension, optimizer, ...
        :param first_only: restore only the first element or all
        :param custom_functions: dictionary holding all custom elements
        :param system_def: dictionary holding all elements including named
        """
        elements = []
        if key in custom_functions.keys() and len(custom_functions[key]) > 0:
            current_custom = list(custom_functions[key].values())
            elements.extend(current_custom)
        if key in system_def:
            elements.extend(system_def[key])

        return elements[0] if first_only else elements

    @staticmethod
    def __load_custom_functions(
        user_path: str,
        custom_type_name: str,
        custom_element_name: str,
    ):
        """
        Load custom function restores a serialized function as executable function in the current context

        :param user_path: user path to the model store
        :param custom_type_name: name of the custom type to recover
        :param custom_element_name: name of the custom element to recover
        """
        # load code object
        with open(
            Trainer.__custom_element_path.format(
                user_path, custom_type_name, custom_element_name
            ),
            "rb",
        ) as reader:
            return dill.load(reader)  # nosec

    def predict(self, **kwargs: Dict[str, Any]) -> tf.Tensor:
        """
        Run a prediction with the current model in the trainer
        and return the result.

        :param kwargs: input as in the fit method by keras/TensorFlow
        :return: prediction result as tensor
        """
        return self._model.predict(**kwargs)
