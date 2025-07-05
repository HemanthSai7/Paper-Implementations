from src.models.layers import BaseLayer

import tensorflow as tf

class ClippedReLU(BaseLayer):
    def __init__(
        self,
        max_value: float = 20.0,
        name: str = "clipped_relu",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.max_value = max_value

    def call(self, inputs):
        return tf.minimum(tf.nn.relu(inputs), self.max_value)