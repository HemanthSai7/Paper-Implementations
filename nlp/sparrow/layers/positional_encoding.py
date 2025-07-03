import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        name="positional_encoding",
        **kwargs,
    ):
        super(PositionalEncoding, self).__init__(trainable=False, name=name, **kwargs)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def encode(self, max_query_length: int, embed_dim: int) -> tf.Tensor:
        pos = tf.expand_dims(tf.range(max_query_length -1, -1, -1.0, dtype=tf.float32), axis=-1)
        index = tf.expand_dims(tf.range(0, embed_dim, dtype=tf.float32), axis=0)

        pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2) / embed_dim)))
        
        sin = tf.expand_dims(tf.sin(pe[:, 0::2]), axis=-1)
        cos = tf.expand_dims(tf.cos(pe[:, 1::2]), axis=-1)

        pe = tf.concat([sin, cos], axis=-1)
        pe = tf.reshape(pe, [max_query_length, embed_dim])

        return tf.expand_dims(pe, axis=0)

    def call(self, inputs):
        _, max_query_length, embed_dim = inputs.shape
        pe = self.encode(max_query_length, embed_dim)
        embeddings = pe + inputs
        embeddings = self.dropout(embeddings)
        return embeddings