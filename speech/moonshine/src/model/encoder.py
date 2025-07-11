from src.utils import math_util
from src.model.layers import MultiHeadAttention, RoPEPositionalEncoding, FFNModule

from typing import Callable, Union

import tensorflow as tf

EPSILON = 1e-6

__all__ = [
    "MoonshineEncoder",
]

@tf.keras.utils.register_keras_serializable(package=__name__)
class Conv1dSubsamplingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_dim: int = 288,
        kernel_size: list = [127, 7, 3],
        strides: list = [64 ,3, 2],
        padding: str = ["same", "same", "same"],
        activations: list = ["tanh", "gelu", "gelu"],
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        dropout_rate: float = 0.0,
        name: str = "conv1d_subsampling",
        **kwargs,
    ):
        super(Conv1dSubsamplingLayer, self).__init__(name=name, **kwargs)
        self.filters = [model_dim, 2 * model_dim, model_dim]
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.do = tf.keras.layers.Dropout(rate=dropout_rate, name=f"{name}_dropout")
        self.ln = tf.keras.layers.LayerNormalization(
                name=f"{name}_ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=tf.float32
        )

        self.conv = []
        for i in range(len(self.kernel_size)):
            conv = tf.keras.layers.Conv1D(
                filters=self.filters[i],
                kernel_size=self.kernel_size[i],
                strides=self.strides[i],
                padding=self.padding[i],
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f"{name}_conv_{i}",
            )
            ac = tf.keras.layers.Activation(activations[i], name=f"{name}_activation_{i + 1}")
            self.conv.append({"conv": conv, "activation": ac})

    def call(self, inputs, training=False):
        inputs, inputs_length = inputs
        inputs = tf.expand_dims(inputs, axis=-1)

        current_outputs = inputs
        current_length = inputs_length
        
        for i, conv_config in enumerate(self.conv):
            current_outputs = conv_config["conv"](current_outputs, training=training)
            current_outputs = conv_config["activation"](current_outputs)

            current_length = math_util.get_conv_length(
                current_length,
                self.kernel_size[i],
                self.strides[i],
                self.padding[i],
            )

        outputs = self.do(current_outputs, training=training)
        outputs = self.ln(outputs)

        outputs_length = tf.cast(current_length, dtype=tf.int32)
        return outputs, outputs_length
    
    def compute_output_shape(self, input_shape):
        # input_shape is a tuple: (tensor_shape, length_shape)
        # tensor_shape is (batch, time, features)
        tensor_shape = input_shape[0]

        batch_dim = tensor_shape[0]
        seq_dim = tensor_shape[1]

        current_seq_dim = seq_dim
        for i in range(len(self.kernel_size)):
            if current_seq_dim is not None:
                current_seq_dim = math_util.get_conv_length_py(
                    current_seq_dim,
                    self.kernel_size[i],
                    self.padding[i],
                    self.strides[i],
                )
            else:
                current_seq_dim = None # Propagate None
                break
        
        output_feature_dim = self.filters[-1]

        return ((batch_dim, current_seq_dim, output_feature_dim), (batch_dim,))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.filters[-1],
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "dropout_rate": self.do.rate,
        })
        return config
    
@tf.keras.utils.register_keras_serializable(package=__name__)
class MHASModule(tf.keras.layers.Layer):
    def __init__(
        self,
        head_size: int,
        num_heads: int,
        dropout: float = 0.0,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "mhsa_module",
        **kwargs,
    ):
        super(MHASModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.mha = MultiHeadAttention(
            name=f"{name}_mha",
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_res_add")

    def call(self, inputs, training=False):
        query, key, outputs = inputs
        outputs = self.mha([query, key, outputs], training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([outputs, inputs[2]])
        outputs = self.ln(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        
        query_shape = input_shape[0] # (batch, seq, dim)
        
        batch_dim = query_shape[0]
        seq_dim = query_shape[1]
        feature_dim = input_shape[2][2] # Feature dimension of the 'outputs' tensor (value)

        return (batch_dim, seq_dim, feature_dim)
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.res_add.get_config())
        return conf

@tf.keras.utils.register_keras_serializable(package=__name__)
class MoonshineEncoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.0,
        fc_factor: int = 4,
        activation: str = "gelu",
        head_size: int = 64,
        num_heads: int = 4,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "encoder_block",
        **kwargs,
    ):
        super(MoonshineEncoderBlock, self).__init__(name=name, **kwargs)

        self.mhsa = MHASModule(
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.ffn = FFNModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )

    def call(self, inputs, training=False, mask=None):
        query_pos_emb, key_pos_emb, outputs = inputs
        outputs = self.mhsa([query_pos_emb, key_pos_emb, outputs], training=training)
        outputs = self.ffn(outputs, training=training)
        outputs = self.ln(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        outputs_shape = input_shape[2] # (batch, seq_len, features)
        return outputs_shape
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.mhsa.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.ffn.get_config())
        return conf

@tf.keras.utils.register_keras_serializable(package=__name__)
class MoonshineEncoder(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int = 6,
        dropout: float = 0.1,
        fc_factor: int = 4,
        activation: str = "gelu",
        head_size: int = 32,
        num_heads: int = 4,
        kernel_regularizer: Union[str, Callable] = None,
        bias_regularizer: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "zeros",
        name: str = "encoder",
        **kwargs,
    ):
        super(MoonshineEncoder, self).__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.conv_subsampling = Conv1dSubsamplingLayer(
            model_dim=input_dim,
            kernel_size=[127, 7, 3],
            strides=[64, 3, 2],
            padding=["same", "same", "same"],
            activations=["tanh", "gelu", "gelu"],
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.rotary_pos_emb = RoPEPositionalEncoding(name=f"{name}_rope_pos_emb")
        self.blocks = [
            MoonshineEncoderBlock(
                input_dim=input_dim,
                dropout=dropout,
                fc_factor=fc_factor,
                activation=activation,
                head_size=head_size,
                num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f"{name}_{i}",
            )
            for i in range(num_blocks)
        ]
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

    def call(self, inputs, training=False, mask=None):
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv_subsampling([outputs, outputs_length], training=training)
        outputs = self.do(outputs, training=training)

        query_pos_emb = self.rotary_pos_emb(outputs)
        key_pos_emb = self.rotary_pos_emb(outputs)
        for block in self.blocks:
            outputs = block([query_pos_emb, key_pos_emb, outputs], training=training, mask=mask)
        outputs = self.ln(outputs)
        return outputs
    
    def compute_output_shape(self, input_shape):
        conv_subsampling_input_shape = input_shape 
        conv_subsampling_output_shapes = self.conv_subsampling.compute_output_shape(conv_subsampling_input_shape)
        
        subsampled_features_shape = conv_subsampling_output_shapes[0]  # (batch, subsampled_time, conv_output_filters)
        final_output_shape = subsampled_features_shape
        return final_output_shape
    
    def get_config(self):
        conf = super().get_config()
        conf.update(self.ln.get_config())
        conf.update(self.rotary_pos_emb.get_config())
        for block in self.blocks:
            conf.update(block.get_config())
        return conf
