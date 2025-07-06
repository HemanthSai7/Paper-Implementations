from src.model.layers import (
    SeparableConv1DSubsamplingLayer, 
    PositionalEncoding, 
    GLU,
    LongAttention
)
from src.utils import shape_util

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class FFModule(tf.keras.layers.Layer):
    r"""Feed forward module
    
    Args:
        input_dim: int: input dimension
        dropout: float: dropout rate
        fc_factor: float: factor to scale the input dimension for the hidden layer
        kernel_regularizer: tf.keras.regularizers.Regularizer: regularizer for kernel
        bias_regularizer: tf.keras.regularizers.Regularizer: regularizer for bias
        kernel_initializer: tf.keras.initializers.Initializer: initializer for kernel
        bias_initializer: tf.keras.initializers.Initializer: initializer for bias
        name: str: name of the layer

    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   fflayer(.)              # 4 * input_dim
      |   swish(.)
      |   dropout(.)
      |   fflayer(.)              # input_dim
      |   dropout(.)
      |   * 1/2
      \   /
        +
        |
      output
    
    """
    def __init__(
            self,
            input_dim,
            dropout = 0.0,
            fc_factor = 0.5,
            kernel_regularizer = None,
            bias_regularizer = None,
            kernel_initializer = None,
            bias_initializer = "zeros",
            name = "ff_module",
            **kwargs,
    ):
        super(FFModule, self).__init__(name = name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            name = f"{name}_ln",
            gamma_regularizer = kernel_regularizer,
            beta_regularizer = kernel_regularizer,
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim,
            name = f"{name}_dense1",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.swish1 = tf.keras.layers.Activation(tf.nn.swish, name = f"{name}_swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim,
            name = f"{name}_dense2",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name = f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name = f"{name}_add")

    def call(
            self,
            inputs,
            training = False,
    ):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish1(outputs)
        outputs = self.do1(outputs, training = training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training = training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        conf = super(FFModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.res_add.get_config())
        return 
    

class MHSAModule(tf.keras.layers.Layer):
    r"""Multi-head self-attention module
    
    Args:
        head_size: int: size of each head
        num_heads: int: number of heads
        dropout: float: dropout rate
        mha_type: str: type of multi-head attention to use
        kernel_regularizer: tf.keras.regularizers.Regularizer: regularizer for kernel
        bias_regularizer: tf.keras.regularizers.Regularizer: regularizer for bias
        kernel_initializer: tf.keras.initializers.Initializer: initializer for kernel
        bias_initializer: tf.keras.initializers.Initializer: initializer for bias
        name: str: name of the layer


    architecture::
    input
    /   \
    |   ln(.)                   # input_dim
    |   mhsa(.)                 # head_size = dmodel // num_heads
    |   dropout(.)
    \   /
        +
        |
    output

    """
    def __init__(
            self,
            head_size,
            num_heads,
            dropout = 0.0,
            mha_type = "relmha",
            kernel_regularizer = None,
            bias_regularizer = None,
            kernel_initializer = None,
            bias_initializer = "zeros",
            name = "mhsa_module",
            **kwargs,
    ):
        super(MHSAModule, self).__init__(name = name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name = f"{name}_ln",
            gamma_regularizer = kernel_regularizer,
            beta_regularizer = kernel_regularizer,
        )
        self.mha = LongAttention(
            name = f"{name}_mhsa",
            head_size = head_size,
            num_heads = num_heads,
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(
            self,
            inputs,
            training = False,
            mask = None,
            **kwargs,
    ):
        # [BS, Seq_len, d_model] -> [BS, Seq_len, d_model]
        inputs, pos = inputs # pos is the positional encoding
        # [BS, Seq_len, d_model] -> [BS, Seq_len, d_model]
        outputs = self.ln(inputs, training=training)
        # [BS, Seq_len, d_model] -> [BS, Seq_len, d_model]
        outputs = self.mha([outputs, outputs, outputs, pos], training=training, mask=mask)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        conf = super(MHSAModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class ConvModule(tf.keras.layers.Layer):
    r"""Convolutional module

    Args:
        input_dim: int: input dimension
        kernel_size: int: kernel size for depthwise convolution
        dropout: float: dropout rate
        scale_factor: int: factor to scale the input dimension for the pointwise convolution
        depth_multiplier: int: depth multiplier for depthwise convolution
        kernel_initializer: tf.keras.initializers.Initializer: initializer for kernel
        bias_initializer: tf.keras.initializers.Initializer: initializer for bias
        kernel_regularizer: tf.keras.regularizers.Regularizer: regularizer for kernel
        bias_regularizer: tf.keras.regularizers.Regularizer: regularizer for bias
        name: str: name of the layer

    architecture::
      input
      /   \
      |   ln(.)                   # input_dim
      |   conv1d(.)              # 2 * input_dim
      |    |
      |   glu(.)                  # input_dim
      |   depthwise_conv_1d(.)
      |   bnorm(.)
      |   swish(.)
      |    |
      |   conv1d(.)
      |   dropout(.)
      \   /
        +
        |
      output
    """
    def __init__(
            self,
            input_dim,
            kernel_size = 32,
            dropout = 0.0,
            scale_factor = 2,
            depth_multiplier = 1,
            kernel_initializer = None,
            bias_initializer = "zeros",
            kernel_regularizer = None,
            bias_regularizer = None,
            name = "conv_module",
            **kwargs,
    ):
        super(ConvModule, self).__init__(name = name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name = f"{name}_ln",
            gamma_regularizer = kernel_regularizer,
            beta_regularizer = kernel_regularizer,
        )
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters = scale_factor * input_dim,
            kernel_size = 1,
            strides = 1,
            padding = "valid",
            name = f"{name}_pw_conv_1",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.glu = GLU(axis=-1, name = f"{name}_glu")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size = (kernel_size, 1),
            depth_multiplier = depth_multiplier,
            strides = 1,
            padding = "same",
            name = f"{name}_dw_conv",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name = f"{name}_bn",
            gamma_regularizer = kernel_regularizer,
            beta_regularizer = kernel_regularizer,
        )
        self.swish = tf.keras.layers.Activation(tf.nn.swish, name = f"{name}_swish_activation")
        self.pw_conv_2 = tf.keras.layers.Conv1D(
            filters = input_dim,
            kernel_size = 1,
            strides = 1,
            padding = "valid",
            name = f"{name}_pw_conv_2",
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name = f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name = f"{name}_add")

    def call(
            self,
            inputs,
            training = False,
            **kwargs
    ):
        outputs = self.ln(inputs, training=training)
        B, T, E = shape_util.shape_list(outputs)
        outputs = tf.reshape(outputs, [B, T, 1, E])
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.reshape(outputs, [B, T, E])
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class ConformerBlock(tf.keras.layers.Layer):
    r"""Conformer block

    Args:
        input_dim: int: input dimension
        dropout: float: dropout rate
        fc_factor: float: factor to scale the input dimension for the hidden layer
        head_size: int: size of each head
        num_heads: int: number of heads
        mha_type: str: type of multi-head attention to use
        kernel_size: int: kernel size for depthwise convolution
        depth_multiplier: int: depth multiplier for depthwise convolution
        kernel_regularizer: tf.keras.regularizers.Regularizer: regularizer for kernel
        bias_regularizer: tf.keras.regularizers.Regularizer: regularizer for bias
        kernel_initializer: tf.keras.initializers.Initializer: initializer for kernel
        bias_initializer: tf.keras.initializers.Initializer: initializer for bias
        name: str: name of the layer
    
    architecture::
      x = x + 1/2 * FFN(x)
      x = x + MHSA(x)
      x = x + Conv(x)
      x = x + 1/2 * FFN(x)
      y = ln(x)
    """
    def __init__(
            self,
            input_dim: int,
            dropout: float = 0.0,
            fc_factor: float = 0.5,
            head_size: int = 36,
            num_heads: int = 4,
            mha_type: str = "relmha",
            kernel_size: int = 32,
            depth_multiplier: int = 1,
            kernel_regularizer = None,
            bias_regularizer = None,
            kernel_initializer = None,
            bias_initializer = "zeros",
            name = "conformer_block",
            **kwargs,
    ):
        super(ConformerBlock, self).__init__(name = name, **kwargs)
        self.ffm1 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name=f"{name}_ff_module1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.mhsam = MHSAModule(
            mha_type=mha_type,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_mhsa_module",
        )
        self.convm = ConvModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_conv_module",
        )
        self.ffm2 = FFModule(
            input_dim=input_dim,
            dropout=dropout,
            fc_factor=fc_factor,
            name=f"{name}_ff_module2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer,
        )

    def call(
            self,
            inputs,
            training = False,
            mask = None,
            **kwargs,
    ):
        inputs, pos = inputs
        outputs = self.ffm1(inputs, training=training, **kwargs)
        outputs = self.mhsam([outputs, pos], training=training, mask=mask, **kwargs)
        outputs = self.convm(outputs, training=training, **kwargs)
        outputs = self.ffm2(outputs, training=training, **kwargs)
        outputs = self.ln(outputs, training=training)
        return outputs
    
    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf
    

class ConformerEncoder(tf.keras.Model):
    def __init__(
            self,
            subsampling: dict,
            dmodel: int = 128,
            num_blocks: int = 16,
            head_size: int = 36,
            num_heads: int = 4,
            mha_type: str = "relmha",
            kernel_size: int = 32,
            depth_multiplier: int = 1,
            fc_factor: float = 0.5,
            dropout: float = 0.0,
            kernel_regularizer = None,
            bias_regularizer = None,
            kernel_initializer = None,
            bias_initializer = "zeros",
            name = "conformer_encoder",
            **kwargs,
    ):
        super(ConformerEncoder, self).__init__(name = name, **kwargs)
        
        self.conv_subsampling = SeparableConv1DSubsamplingLayer(
            filters=subsampling.filters,
            kernel_size=subsampling.kernel_size,
            strides=subsampling.strides,
            name=f"{name}_subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )

        self.pe = PositionalEncoding(name=f"{name}_pe")
        
        self.linear = tf.keras.layers.Dense(
            dmodel,
            name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name=f"{name}_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)

    def call(
            self,
            inputs,
            training = False,
            mask = None,
            **kwargs,
    ):
        # inputs with shape [BS, T, V1, V2]
        outputs, outputs_length = inputs
        outputs, outputs_length = self.conv_subsampling([outputs, outputs_length], training=training)
        outputs = self.linear(outputs, training=training)
        pe = self.pe(outputs)
        outputs = self.do(outputs, training=training)
        for cblock in self.conformer_blocks:
            outputs = cblock([outputs, pe], training=training, mask=mask, **kwargs)
        return outputs, outputs_length
    
    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        conf.update(self.pe.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf
