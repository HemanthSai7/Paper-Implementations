import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class OutputLayer(layers.Layer):
    """
    QANet Output Layer: predicts start and end positions for answer span.
    Reference: QANet paper (Yu et al. 2018)
    """
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing OutputLayer: hidden_dim={hidden_dim}")
        self.start_dense = layers.Dense(1)
        self.end_dense = layers.Dense(1)

    def call(self, M0, M1, M2, mask=None):
        tf.get_logger().debug(f"OutputLayer.call M0 shape: {M0.shape}, M1 shape: {M1.shape}, M2 shape: {M2.shape}")
        # M0, M1, M2: outputs from model encoder blocks (see QANet architecture)
        # Compute logits for start and end positions
        x_start = tf.concat([M0, M1], axis=-1)
        x_end = tf.concat([M0, M2], axis=-1)
        logits_start = tf.squeeze(self.start_dense(x_start), axis=-1)  # (batch, seq_len)
        logits_end = tf.squeeze(self.end_dense(x_end), axis=-1)   
        print(mask is not None)
        ff     # (batch, seq_len)
        if mask is not None:
            logits_start += (1.0 - tf.cast(mask, tf.float32)) * -1e9
            logits_end += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        prob_start = tf.nn.softmax(logits_start, axis=-1)
        prob_end = tf.nn.softmax(logits_end, axis=-1)
        tf.get_logger().debug(f"OutputLayer.call prob_start shape: {prob_start.shape}, prob_end shape: {prob_end.shape}")
        return prob_start, prob_end

    def compute_output_shape(self, input_shape):
        tf.get_logger().debug(f"OutputLayer.compute_output_shape input: {input_shape}")
        m0_shape = tf.TensorShape(input_shape[0]).as_list()
        if m0_shape is None or len(m0_shape) < 2:
            out_shape = (tf.TensorShape([None, None]), tf.TensorShape([None, None]))
        else:
            batch_size, seq_len = m0_shape[0], m0_shape[1]
            out_shape = (tf.TensorShape([batch_size, seq_len]), tf.TensorShape([batch_size, seq_len]))
        tf.get_logger().debug(f"OutputLayer.compute_output_shape output: {out_shape}")
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.start_dense.units + self.end_dense.units})
        return config
