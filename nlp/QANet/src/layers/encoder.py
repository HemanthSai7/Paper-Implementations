import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class DepthwiseSeparableConv(tf.keras.layers.Layer):
    def __init__(
        self, 
        num_filters: int, 
        kernel_size: int, 
        padding: str = "same",
        name: str = "depthwise_separable_conv",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.depthwise_conv = tf.keras.layers.SeparableConv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            activation="relu",
        )
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        outputs = self.ln(inputs)
        outputs = self.depthwise_conv(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.depthwise_conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.depthwise_conv.filters,
            'kernel_size': self.depthwise_conv.kernel_size
        })
        return config


@tf.keras.utils.register_keras_serializable(package=__name__)
class FFNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        name: str = "ffn",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ffn1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.do = tf.keras.layers.Dropout(dropout_rate)
        self.ffn2 = tf.keras.layers.Dense(output_dim)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        outputs = self.ffn1(inputs)
        outputs = self.do(outputs, training=training)
        outputs = self.ffn2(outputs)
        outputs = self.ln(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.ffn2.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.ffn1.units,
            "output_dim": self.ffn2.units,
            "dropout_rate": self.dropout.rate,
        })

    
@tf.keras.utils.register_keras_serializable(package=__name__)
class EmbeddingEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_conv_layers: int,
        num_filters: int,
        kernel_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float = 0.1,
        project_input: bool = True,
        name: str = "embedding_encoder_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.project_input = project_input
        if self.project_input:
            self.input_projection = tf.keras.layers.Dense(num_filters)
        else:
            self.input_projection = None
        self.conv_layers = []

        for i in range(num_conv_layers):
            dscnn = DepthwiseSeparableConv(
                num_filters=num_filters,
                kernel_size=kernel_size,
                name=f"dscnn_{i}"
            )
            self.conv_layers.append({"dscnn": dscnn})
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=num_filters,
            dropout=dropout_rate,
        )
        self.ffn = FFNetwork(
            hidden_dim=ffn_dim,
            output_dim=num_filters,
            dropout_rate=dropout_rate,
            name="ffn"
        )
        self.do = tf.keras.layers.Dropout(dropout_rate, name="dropout")
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm")

    def positional_embedding(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[-1]
        pos = tf.expand_dims(tf.range(seq_len, dtype=tf.float32), axis=-1)
        index = tf.expand_dims(tf.range(d_model, dtype=tf.float32), axis=0)
        angle_rates = 1.0 / tf.pow(10000.0, (2 * (index // 2)) / tf.cast(d_model, tf.float32))
        rad = pos * angle_rates
        sin = tf.sin(rad[:, 0::2])
        cos = tf.cos(rad[:, 1::2])
        pos_encoding = tf.expand_dims(tf.concat([sin, cos], axis=-1), axis=0)
        return inputs + tf.cast(pos_encoding, inputs.dtype)


    def call(self, inputs, training=False, mask=None):
        if self.project_input and self.input_projection is not None:
            outputs = self.input_projection(inputs)
        else:
            outputs = inputs

        outputs = self.positional_embedding(outputs)
        for conv in self.conv_layers:
            residual = outputs
            outputs = conv["dscnn"](outputs)
            outputs = self.do(outputs, training=training)
            outputs = outputs + residual

        residual = outputs
        outputs = self.ln(outputs)
        outputs = self.self_attn(outputs, outputs, attention_mask=mask, training=training)
        outputs = self.do(outputs, training=training)
        outputs = outputs + residual

        residual = outputs
        outputs = self.ln(outputs)
        outputs = self.ffn(outputs)
        outputs = self.do(outputs, training=training)
        outputs = outputs + residual
        outputs = self.ln(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.conv_layers[0]['dscnn'].depthwise_conv.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_conv_layers': len(self.conv_layers),
            'num_filters': self.conv_layers[0]['dscnn'].depthwise_conv.filters,
            'kernel_size': self.conv_layers[0]['dscnn'].depthwise_conv.kernel_size,
            'num_heads': self.self_attn.num_heads,
            'ffn_dim': self.ffn.dense1.units,
            'dropout_rate': self.dropout.rate,
            'project_input': self.project_input
        })
        return config


@tf.keras.utils.register_keras_serializable(package=__name__)
class QANetEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        num_conv_layers: int,
        num_filters: int,
        kernel_size: int,
        ffn_dim: int,
        dropout_rate: float = 0.1,
        name: str = "qanet_encoder_block",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
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

    def call(self, inputs, training=False, mask=None):
        for block in self.blocks:
            outputs = block(inputs, training=training, mask=mask)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.blocks[0].conv_layers[0]['dscnn'].depthwise_conv.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_blocks': len(self.blocks),
            'num_conv_layers': len(self.blocks[0].conv_layers),
            'num_filters': self.blocks[0].conv_layers[0]['dscnn'].depthwise_conv.filters,
            'kernel_size': self.blocks[0].conv_layers[0]['dscnn'].depthwise_conv.kernel_size,
            'num_heads': self.blocks[0].self_attn.num_heads,
            'ffn_dim': self.blocks[0].ffn.dense1.units,
            'dropout_rate': self.blocks[0].dropout.rate
        })
        return config