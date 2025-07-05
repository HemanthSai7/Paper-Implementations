import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class ContextQueryAttention(layers.Layer):
    """
    Implements the Context-to-Query and Query-to-Context Attention as in QANet.
    Reference: QANet paper (Yu et al. 2018)
    """
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing ContextQueryAttention: hidden_dim={hidden_dim}")
        tf.get_logger().info(f"Initializing ContextQueryAttention: config={kwargs}")
        self.hidden_dim = hidden_dim
        self.similarity_dense = layers.Dense(1, use_bias=False)

    def call(self, context, query, context_mask=None, query_mask=None):
        tf.get_logger().debug(f"ContextQueryAttention.call context shape: {context.shape}, query shape: {query.shape}")
        # context: (batch, c_len, dim)
        # query: (batch, q_len, dim)
        batch_size = tf.shape(context)[0]
        c_len = tf.shape(context)[1]
        q_len = tf.shape(query)[1]
        dim = tf.shape(context)[2]

        print(context.shape, query.shape)

        # Expand dims for broadcasting
        context_exp = tf.expand_dims(context, 2)  # (batch, c_len, 1, dim)
        query_exp = tf.expand_dims(query, 1)      # (batch, 1, q_len, dim)
        # Tile for broadcasting
        context_tiled = tf.tile(context_exp, [1, 1, q_len, 1])  # (batch, c_len, q_len, dim)
        query_tiled = tf.tile(query_exp, [1, c_len, 1, 1])      # (batch, c_len, q_len, dim)
        # Elementwise multiplication
        elem_mult = context_tiled * query_tiled    
        print(elem_mult.shape)             # (batch, c_len, q_len, dim)
        # Concatenate along last axis
        concat = tf.concat([context_tiled, query_tiled, elem_mult], axis=-1)  # (batch, c_len, q_len, 3*dim)
        # Similarity matrix S
        S = tf.squeeze(self.similarity_dense(concat), axis=-1)  # (batch, c_len, q_len)

        if query_mask is not None:
            S += (1.0 - tf.cast(tf.expand_dims(query_mask, 1), tf.float32)) * -1e9
        # Context-to-query attention
        a = tf.nn.softmax(S, axis=2)  # (batch, c_len, q_len)
        c2q = tf.matmul(a, query)     # (batch, c_len, dim)

        # Query-to-context attention
        b = tf.nn.softmax(tf.reduce_max(S, axis=2), axis=1)  # (batch, c_len)
        q2c = tf.matmul(tf.expand_dims(b, 1), context)       # (batch, 1, dim)
        q2c = tf.tile(q2c, [1, c_len, 1])                   # (batch, c_len, dim)

        # Final output: [context; c2q; context * c2q; context * q2c]
        output = tf.concat([context, c2q, context * c2q, context * q2c], axis=-1)
        tf.get_logger().debug(f"ContextQueryAttention.call output shape: {output.shape}")
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
        config.update({'hidden_dim': self.hidden_dim})
        return config
