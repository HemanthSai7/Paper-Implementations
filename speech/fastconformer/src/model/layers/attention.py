import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class LongAttention(tf.keras.layers.Layer):
    ...