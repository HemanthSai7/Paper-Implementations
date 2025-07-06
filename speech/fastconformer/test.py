from src.model.layers.subsampling import SeparableConv1DSubsamplingLayer

import tensorflow as tf

subsampling_config = {
    "kernel_size": [127, 7, 3],
    "strides": [64, 3, 3],
    "padding": ["same", "same", "same"],
    "activation": ["tanh", "gelu", "gelu"]
}

layer = SeparableConv1DSubsamplingLayer(
    model_dim=288,
    subsampling_config=subsampling_config,
    kernel_regularizer=None,
    bias_regularizer=None,
    name="conv1d_subsampling"
)

inputs = tf.random.normal((2, 1000, 80, 1))
print("Inputs shape:", inputs.shape)
outputs, outputs_length = layer((inputs, tf.shape(inputs)[1]))

batch_size = tf.shape(inputs)[0]
actual_lengths = tf.fill([batch_size], 1000)

mask = layer.compute_mask((inputs, actual_lengths))

print("Outputs shape:", outputs.shape)
print("Outputs length:", outputs_length)
print("Mask shape:", mask)