import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class CharEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, 
        char_vocab_size: int, 
        char_embedding_dim: int, 
        num_filters: int, 
        kernel_size: int, 
        padding: str = "same",
        kernel_intializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        trainable: bool = True,
        name="char_embedding",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_filters = num_filters
        self.char_embedding = tf.keras.layers.Embedding(
            input_dim=char_vocab_size,
            output_dim=char_embedding_dim,
            trainable=trainable,
            name="char_embedding_layer"
        )
        self.conv = tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding=padding,
            kernel_initializer=kernel_intializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="char_conv_layer"
        )
        self.pool = tf.keras.layers.GlobalMaxPooling1D(name="char_pool_layer")

    def call(self, inputs, training=False, **kwargs):
        #inputs -> (B, seq_len, word_len)
        batch_size, seq_length, word_length = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        outputs = tf.reshape(inputs, [-1, word_length])
        outputs = self.char_embedding(outputs, training=training)
        outputs = self.conv(outputs)
        outputs = self.pool(outputs)
        outputs = tf.reshape(outputs, [batch_size, seq_length, self.num_filters]) # (B, seq_len, num_filters)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.conv.filters)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "char_vocab_size": self.char_vocab_size,
            "char_embedding_dim": self.char_embedding_dim,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "trainable": self.char_embedding.trainable
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class WordEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        word_vocab_size: int,
        word_embedding_dim: int,
        pretrained_embeddings = None,
        trainable: bool = False,
        mask_zero: bool = False,
        name = "word_embedding",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if pretrained_embeddings is not None:
            self.word_embedding = tf.keras.layers.Embedding(
                input_dim=word_vocab_size,
                output_dim=word_embedding_dim,
                trainable=trainable,
                weights=[pretrained_embeddings],
                mask_zero=mask_zero,
                name="word_embedding_layer"
            )
        else:
            self.word_embedding = tf.keras.layers.Embedding(
                input_dim=word_vocab_size,
                output_dim=word_embedding_dim,
                trainable=trainable,
                mask_zero=mask_zero,
                name="word_embedding_layer"
            )

    def call(self, inputs, training=False, **kwargs):
        return self.word_embedding(inputs, training=training)
    
    def compute_output_shape(self, input_shape):
        return self.word_embedding.compute_output_shape(input_shape)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "word_vocab_size": self.word_vocab_size,
            "word_embedding_dim": self.word_embedding_dim,
            "trainable": self.trainable,
            "mask_zero": self.mask_zero
        })
        return config