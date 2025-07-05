import tensorflow as tf


class QANetDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        name: str = "qanet_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.start_dense = tf.keras.layers.Dense(1)
        self.end_dense = tf.keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        # mask = None
        M0, M1, M2 = inputs
        x_start = tf.concat([M0, M1], axis=-1)
        x_end = tf.concat([M0, M2], axis=-1)
        logits_start = tf.squeeze(self.start_dense(x_start),axis=-1)
        logits_end = tf.squeeze(self.end_dense(x_end),axis=-1)

        if mask is not None and tf.is_tensor(mask):
            logits_start += (1.0 - tf.cast(mask, tf.float32)) * -1e9
            logits_end += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        
        prob_start = tf.nn.softmax(logits_start, axis=-1)
        prob_end = tf.nn.softmax(logits_end,axis=-1)

        return prob_start, prob_end

    def compute_output_shape(self, input_shape):
        m0_shape = tf.TensorShape(input_shape[0]).as_list()
        if m0_shape is None or len(m0_shape) < 2:
            out_shape = (tf.TensorShape([None, None]), tf.TensorShape([None, None]))
        else:
            batch_size, seq_len = m0_shape[0], m0_shape[1]
            out_shape = (tf.TensorShape([batch_size, seq_len]), tf.TensorShape([batch_size, seq_len]))
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.start_dense.units + self.end_dense.units})
        return config