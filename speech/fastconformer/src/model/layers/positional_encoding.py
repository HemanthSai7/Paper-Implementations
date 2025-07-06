from src.utils.shape_util import shape_list

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding layer"""
    def __init__(
            self,
            name: str = "positional_encoding",
            **kwargs,
    ):
        super(PositionalEncoding, self).__init__(trainable=False, name=name, **kwargs)

    def build(
            self,
            input_shape,
    ):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    @staticmethod
    def encode(
        max_len,
        dmodel,
    ):
        """Encode position with sine and cosine function

        Args:
            max_len: int: maximum length of the position
            dmodel: int: dimension of model

        Returns:
            positional encoding: tf.Tensor: [1, max_len, dmodel]

        Formulae:
            PE(pos, 2i) = sin(pos / 10000^(2i/dmodel)) 
            PE(pos, 2i+1) = cos(pos / 10000^(2i/dmodel))

        """
        pos = tf.expand_dims(tf.range(max_len -1, -1, -1.0, dtype=tf.float32), axis=1) # [max_len, 1]
        index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0) # [1, dmodel]

        pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2) / dmodel))) # [max_len, dmodel]

        # Sin cos will be [max_len, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.expand_dims(tf.sin(pe[:, 0::2]), axis=-1)
        cos = tf.expand_dims(tf.cos(pe[:, 1::2]), axis=-1)

        # Then add sin and cos, which results in [time, size]
        pe = tf.concat([sin, cos], axis=-1)
        pe = tf.reshape(pe, [max_len, dmodel])

        return tf.expand_dims(pe, axis=0) # [1, time, size]
    
    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, max_len, dmodel = shape_list(inputs)
        pe = self.encode(max_len, dmodel)
        return tf.cast(pe, dtype=inputs.dtype)
    
    def get_config(self):
        conf = super().get_config()
        return conf