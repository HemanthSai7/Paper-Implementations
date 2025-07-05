import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package=__name__)
class WordEmbedding(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, trainable=False, mask_zero=False, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing WordEmbedding: vocab_size={vocab_size}, embedding_dim={embedding_dim}, trainable={trainable}, mask_zero={mask_zero}")
        if pretrained_embeddings is not None:
            self.embedding = layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[pretrained_embeddings],
                trainable=trainable,
                mask_zero=mask_zero
            )
        else:
            self.embedding = layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                trainable=trainable,
                mask_zero=mask_zero
            )

    def call(self, inputs):
        tf.get_logger().debug(f"WordEmbedding.call input shape: {inputs.shape}")
        out = self.embedding(inputs)
        tf.get_logger().debug(f"WordEmbedding.call output shape: {out.shape}")
        return out

    def compute_output_shape(self, input_shape):
        tf.get_logger().debug(f"WordEmbedding.compute_output_shape input: {input_shape}")
        out_shape = self.embedding.compute_output_shape(input_shape)
        tf.get_logger().debug(f"WordEmbedding.compute_output_shape output: {out_shape}")
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.embedding.input_dim,
            'embedding_dim': self.embedding.output_dim,
            'trainable': self.embedding.trainable,
            'mask_zero': self.embedding.mask_zero
        })
        return config

@tf.keras.utils.register_keras_serializable(package=__name__)
class CharEmbedding(layers.Layer):
    def __init__(self, char_vocab_size, char_embedding_dim, num_filters, kernel_size, trainable=True, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing CharEmbedding: char_vocab_size={char_vocab_size}, char_embedding_dim={char_embedding_dim}, num_filters={num_filters}, kernel_size={kernel_size}, trainable={trainable}")
        self.char_embedding = layers.Embedding(
            input_dim=char_vocab_size,
            output_dim=char_embedding_dim,
            trainable=trainable
        )
        self.conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same'
        )
        self.pool = layers.GlobalMaxPooling1D()

    def call(self, inputs):
        tf.get_logger().debug(f"CharEmbedding.call input shape: {inputs.shape}")
        # inputs: (batch, seq_len, word_len)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        word_len = tf.shape(inputs)[2]
        x = tf.reshape(inputs, [-1, word_len])  # (batch*seq_len, word_len)
        x = self.char_embedding(x)              # (batch*seq_len, word_len, char_emb_dim)
        x = self.conv(x)                        # (batch*seq_len, word_len, num_filters)
        x = self.pool(x)                        # (batch*seq_len, num_filters)
        x = tf.reshape(x, [batch_size, seq_len, self.conv.filters])  # (batch, seq_len, num_filters)
        tf.get_logger().debug(f"CharEmbedding.call output shape: {x.shape}")
        return x

    def compute_output_shape(self, input_shape):
        tf.get_logger().debug(f"CharEmbedding.compute_output_shape input: {input_shape}")
        out_shape = (input_shape[0], input_shape[1], self.conv.filters)
        tf.get_logger().debug(f"CharEmbedding.compute_output_shape output: {out_shape}")
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'char_vocab_size': self.char_embedding.input_dim,
            'char_embedding_dim': self.char_embedding.output_dim,
            'num_filters': self.conv.filters,
            'kernel_size': self.conv.kernel_size,
            'trainable': self.char_embedding.trainable
        })
        return config
