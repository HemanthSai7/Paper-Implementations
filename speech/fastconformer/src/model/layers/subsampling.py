from typing import Optional
from src.utils.math_util import get_conv_length, merge_two_last_dims

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class SeparableConv1DSubsamplingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_dim: int = 288,
        subsampling_config: dict = None,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        name: str = "conv1d_subsampling",
        **kwargs,
    ):
        super(SeparableConv1DSubsamplingLayer, self).__init__(name=name, **kwargs)
        self.filters = [model_dim, 2 * model_dim, model_dim]
        # self.filters = [model_dim, model_dim]
        self.kernel_size = subsampling_config.get("kernel_size", [127, 7, 3])
        self.strides = subsampling_config.get("strides", [64, 3, 3])
        self.padding = subsampling_config.get("padding", ["same", "same", "same"])
        self.activations = subsampling_config.get("activation", ["tanh", "gelu", "gelu"])

        if len(self.kernel_size) != len(self.strides) or len(self.kernel_size) != len(self.padding) or len(self.kernel_size) != len(self.activations):
            raise ValueError("kernel_size, strides, padding, and activation must have the same length.")

        self.convs = []
        for i in range(len(self.kernel_size)):
            subblock = tf.keras.Sequential(name=f"{name}_subblock_{i+1}")
            subblock.add(
                tf.keras.layers.SeparableConv1D(
                    filters=self.filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.strides[i],
                    padding=self.padding[i],
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f"{name}_conv_{i+1}"
                )
            )
            subblock.add(tf.keras.layers.Activation(self.activations[i], name=f"{name}_activation_{i+1}"))
            self.convs.append(subblock)

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = merge_two_last_dims(outputs)
        for block in self.convs:
            outputs = block(outputs, training=training)
            outputs_length = get_conv_length(
                outputs_length,
                kernel_size=block.layers[0].kernel_size[0],
                padding=block.layers[0].padding,
                strides=block.layers[0].strides[0]
            )
        return outputs, outputs_length
    
    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        for block in self.convs:
            maxlen, outputs_length = (
                get_conv_length(
                    length, kernel_size=block.layers[0].kernel_size[0], padding=block.layers[0].padding, strides=block.layers[0].strides[0]
                )
                for length in (maxlen, outputs_length)
            )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    
    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = output_shape[:2] + (output_shape[2] * output_shape[3],)
        for block in self.convs:
            output_shape = block.layers[0].compute_output_shape(output_shape)
        return output_shape, output_length_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.filters[-1],
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "strides": self.strides,
            "padding": self.padding,
            "name": self.name,
            "activations": self.activations,
        })
        return config
