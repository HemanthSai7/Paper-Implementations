import tensorflow as tf
from src.utils import math_util

__all__ = [
    'BaseLayer',
    'Reshape',
    'Identity',
]


@tf.keras.utils.register_keras_serializable(package=__name__)
class BaseLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            trainable=True,
            name=None,
            dtype=None,
            **kwargs,
    ):
        super(BaseLayer, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.supports_masking = True


@tf.keras.utils.register_keras_serializable(package=__name__)
class Reshape(BaseLayer):
    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = math_util.merge_two_last_dims(outputs)
        return outputs, outputs_length
    
    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = output_shape[:2] + (output_shape[2] * output_shape[3],)
        return output_shape, output_length_shape
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class Identity(BaseLayer):
    def call(self, inputs):
        return inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
