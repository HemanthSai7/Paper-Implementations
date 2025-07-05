import tensorflow as tf


class ContextQueryAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        name: str = "context_query_attention",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.similarity_matrix = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs, training=False, context_mask=None, query_mask=None):
        context, query = inputs

        batch_size = tf.shape(context)[0]
        context_len = tf.shape(context)[1]
        query_len = tf.shape(query)[1]
        dim = tf.shape(context)[2]

        context_broadcast = tf.expand_dims(context, 2)
        query_broadcast = tf.expand_dims(query, 1)

        context_broadcast = tf.tile(context_broadcast, [1, 1, query_len, 1])
        query_broadcast = tf.tile(query_broadcast, [1, context_len, 1, 1])

        similarity = context_broadcast * query_broadcast
        similarity = tf.concat([context_broadcast, query_broadcast, similarity], axis=-1)
        S = tf.squeeze(self.similarity_matrix(similarity), axis=-1)

        if query_mask is not None:
            S += (1.0 - tf.cast(tf.expand_dims(query_mask, 1), tf.float32)) * -1e9

        c2q_attention_scores = tf.nn.softmax(S, axis=2)
        c2q = tf.matmul(c2q_attention_scores, query)

        q2c_attention_scores = tf.nn.softmax(tf.reduce_max(S, axis=2), axis=1)
        q2c = tf.matmul(tf.expand_dims(q2c_attention_scores, 1), context)
        q2c = tf.tile(q2c, [1, context_len, 1])

        output = tf.concat([context, c2q, context * c2q, context * q2c], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        context_shape = tf.TensorShape(input_shape[0]).as_list()
        if context_shape is None or len(context_shape) < 3:
            out_shape = tf.TensorShape([None, None, None])
        else:
            out_shape = tf.TensorShape([context_shape[0], context_shape[1], context_shape[2] * 4])
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim
        })
        return config