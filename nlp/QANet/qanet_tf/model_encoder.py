import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package=__name__)
class DepthwiseSeparableConv(layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing DepthwiseSeparableConv: num_filters={num_filters}, kernel_size={kernel_size}")
        self.depthwise_conv = layers.SeparableConv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu'
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
    def call(self, x):
        out = self.depthwise_conv(x)
        out = self.layer_norm(out)
        return out

    def compute_output_shape(self, input_shape):
        out_shape = self.depthwise_conv.compute_output_shape(input_shape)
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.depthwise_conv.filters,
            'kernel_size': self.depthwise_conv.kernel_size
        })
        return config

@tf.keras.utils.register_keras_serializable(package=__name__)
class FeedForwardNetwork(layers.Layer):
    def __init__(self, hidden_dim, output_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing FeedForwardNetwork: hidden_dim={hidden_dim}, output_dim={output_dim}, dropout_rate={dropout_rate}")
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(output_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
    def call(self, x, training=False):
        tf.get_logger().debug(f"FeedForwardNetwork.call input shape: {x.shape}")
        out = self.dense1(x)
        out = self.dropout(out, training=training)
        out = self.dense2(out)
        out = self.layer_norm(out)
        tf.get_logger().debug(f"FeedForwardNetwork.call output shape: {out.shape}")
        return out

    def compute_output_shape(self, input_shape):
        tf.get_logger().debug(f"FeedForwardNetwork.compute_output_shape input: {input_shape}")
        out_shape = self.dense2.compute_output_shape(input_shape)
        tf.get_logger().debug(f"FeedForwardNetwork.compute_output_shape output: {out_shape}")
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.dense1.units,
            'output_dim': self.dense2.units,
            'dropout_rate': self.dropout.rate
        })
        tf.get_logger().debug(f"FeedForwardNetwork.get_config: {config}")
        return config

@tf.keras.utils.register_keras_serializable(package=__name__)
class EmbeddingEncoderBlock(layers.Layer):
    def __init__(self, num_conv_layers, num_filters, kernel_size, num_heads, ffn_dim, dropout_rate=0.1, project_input=True, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing EmbeddingEncoderBlock: num_conv_layers={num_conv_layers}, num_filters={num_filters}, kernel_size={kernel_size}, num_heads={num_heads}, ffn_dim={ffn_dim}, dropout_rate={dropout_rate}")
        self.project_input = project_input
        if self.project_input:
            self.input_projection = layers.Dense(num_filters)
        else:
            self.input_projection = None
        self.conv_layers = [DepthwiseSeparableConv(num_filters, kernel_size) for _ in range(num_conv_layers)]
        self.self_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_filters, dropout=dropout_rate)
        self.ffn = FeedForwardNetwork(ffn_dim, num_filters, dropout_rate)
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

        print(self.conv_layers)
    def call(self, x, training=False, mask=None):
        tf.get_logger().debug(f"EmbeddingEncoderBlock.call input shape: {x.shape}")
        # Project input to num_filters for residual compatibility (only if needed)
        if self.project_input and self.input_projection is not None:
            x = self.input_projection(x)
        # Positional encoding (QANet uses sinusoidal)
        x = self.add_positional_encoding(x)
        # Convolutional sublayers
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            x = self.dropout(x, training=training)
            x = x + residual
        # Self-attention sublayer
        residual = x
        x = self.self_attn(x, x, attention_mask=mask, training=training)
        x = self.dropout(x, training=training)
        x = x + residual
        # Feed-forward sublayer
        residual = x
        x = self.ffn(x, training=training)
        x = self.dropout(x, training=training)
        x = x + residual
        x = self.layer_norm(x)
        tf.get_logger().debug(f"EmbeddingEncoderBlock.call output shape: {x.shape}")
        return x

    def compute_output_shape(self, input_shape):
        tf.get_logger().debug(f"EmbeddingEncoderBlock.compute_output_shape input: {input_shape}")
        # Output shape: (batch, seq_len, num_filters)
        out_shape = (input_shape[0], input_shape[1], self.conv_layers[0].depthwise_conv.filters)
        tf.get_logger().debug(f"EmbeddingEncoderBlock.compute_output_shape output: {out_shape}")
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_conv_layers': len(self.conv_layers),
            'num_filters': self.conv_layers[0].depthwise_conv.filters,
            'kernel_size': self.conv_layers[0].depthwise_conv.kernel_size,
            'num_heads': self.self_attn.num_heads,
            'ffn_dim': self.ffn.dense1.units,
            'dropout_rate': self.dropout.rate,
            'project_input': self.project_input
        })
        tf.get_logger().debug(f"EmbeddingEncoderBlock.get_config: {config}")
        return config
    def add_positional_encoding(self, x):
        # Sinusoidal positional encoding
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[-1]
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        # apply sin to even indices in the array; cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, coses], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return x + tf.cast(pos_encoding, x.dtype)

@tf.keras.utils.register_keras_serializable(package=__name__)
class ModelEncoderBlock(layers.Layer):
    """
    Stacks several EmbeddingEncoderBlocks as in QANet's Model Encoder.
    """
    def __init__(self, num_blocks, num_conv_layers, num_filters, kernel_size, num_heads, ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        tf.get_logger().debug(f"Initializing ModelEncoderBlock: num_blocks={num_blocks}, num_conv_layers={num_conv_layers}, num_filters={num_filters}, kernel_size={kernel_size}, num_heads={num_heads}, ffn_dim={ffn_dim}, dropout_rate={dropout_rate}")
        self.blocks = [
            EmbeddingEncoderBlock(
                num_conv_layers=num_conv_layers,
                num_filters=num_filters,
                kernel_size=kernel_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout_rate=dropout_rate,
                project_input=False
            ) for _ in range(num_blocks)
        ]

    def call(self, x, training=False, mask=None):
        tf.get_logger().debug(f"ModelEncoderBlock.call input shape: {x.shape}")
        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        tf.get_logger().debug(f"ModelEncoderBlock.call output shape: {x.shape}")
        return x

    def compute_output_shape(self, input_shape):
        tf.get_logger().debug(f"ModelEncoderBlock.compute_output_shape input: {input_shape}")
        # Output shape: (batch, seq_len, num_filters)
        out_shape = (input_shape[0], input_shape[1], self.blocks[0].conv_layers[0].depthwise_conv.filters)
        tf.get_logger().debug(f"ModelEncoderBlock.compute_output_shape output: {out_shape}")
        return out_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_blocks': len(self.blocks),
            'num_conv_layers': len(self.blocks[0].conv_layers),
            'num_filters': self.blocks[0].conv_layers[0].depthwise_conv.filters,
            'kernel_size': self.blocks[0].conv_layers[0].depthwise_conv.kernel_size,
            'num_heads': self.blocks[0].self_attn.num_heads,
            'ffn_dim': self.blocks[0].ffn.dense1.units,
            'dropout_rate': self.blocks[0].dropout.rate
        })
        return config
