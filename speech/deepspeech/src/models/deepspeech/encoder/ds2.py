from src.utils import layer_util, math_util
from src.models.layers import BaseLayer, Identity, Reshape

import tensorflow as tf

__all__ = [
    "DeepSpeech2Encoder"
]

@tf.keras.utils.register_keras_serializable(package=__name__)
class ConvBlock(BaseLayer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [11, 41],
        strides: list = [2, 2],
        filters: int = 32,
        padding: str = "causal",
        activation: str = "clipped_relu",
        dropout: float = 0.1,
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        initializer: str = None,
        name: str = "conv_block",
        **kwargs,
    ):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        self.conv = layer_util.get_conv(conv_type)(
            filters = filters,
            kernel_size=kernels,
            strides=strides,
            padding=padding,
            name=conv_type,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            dtype=self.dtype,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="batch_norm", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.act = layer_util.get_activation(name=activation)
        self.do = tf.keras.layers.Dropout(dropout)
        self.time_reduction_factor = self.conv.strides[0]

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.act(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs_length = math_util.conv_output_length(
            outputs_length, filter_size=self.conv.kernel_size[0], stride=self.conv.strides[0], padding=self.conv.padding
        )
        return outputs, outputs_length
    
    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (
            math_util.conv_output_length(
                length, filter_size=self.conv.kernel_size[0], padding=self.conv.padding, stride=self.conv.strides[0])
                for length in (maxlen, outputs_length)
            )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.conv.compute_output_shape(output_shape)
        output_shape = self.bn.compute_output_shape(output_shape)
        output_shape = self.act.compute_output_shape(output_shape)
        output_shape = self.do.compute_output_shape(output_shape)
        return output_shape, output_length_shape


@tf.keras.utils.register_keras_serializable(package=__name__)
class ConvModule(BaseLayer):
    def __init__(
        self,
        conv_type: str = "conv2d",
        kernels: list = [[11, 41], [11, 21], [11, 21]],
        strides: list = [[2, 2], [1, 2], [1, 2]],
        filters: list = [32, 32, 96],
        padding: str = "causal",
        activation: str = "clipped_relu",
        dropout: float = 0.1,
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        initializer: str = None,
        name: str = "conv_module",
        **kwargs,
    ):
        super(ConvModule, self).__init__(name=name, **kwargs)
        assert conv_type in ["conv1d", "conv2d"]
        assert len(kernels) == len(strides) == len(filters)
        assert dropout >=0.0

        self.pre = Reshape(name="preprocess", dtype=self.dtype) if conv_type == "conv1d" else Identity(name="conv_subsampling", dtype=self.dtype)

        self.convs = []
        self.time_reduction_factor = 1
        for i in range(len(filters)):
            conv_block = ConvBlock(
                conv_type=conv_type,
                kernels=kernels[i],
                strides=strides[i],
                filters=filters[i],
                padding=padding,
                activation=activation,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                initializer=initializer,
                name=f"conv_block_{i}",
                dtype=self.dtype,
            )
            self.convs.append(conv_block)
            self.time_reduction_factor *= conv_block.time_reduction_factor

        self.post = Reshape(name="postprocess", dtype=self.dtype) if conv_type == "conv2d" else Identity(name="conv_subsampling", dtype=self.dtype)

    def call(self, inputs, training=False):
        outputs = self.pre(inputs, training=training)
        for conv in self.convs:
            outputs = conv(outputs, training=training)
        outputs = self.post(outputs, training=training)
        return outputs
    
    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        for conv in self.convs:
            maxlen, outputs_length = (
                math_util.conv_output_length(
                    length, filter_size=conv.conv.kernel_size[0], padding=conv.conv.padding, stride=conv.conv.strides[0])
                    for length in (maxlen, outputs_length)
                )
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape = self.pre.compute_output_shape(output_shape)
        for conv in self.convs:
            output_shape = conv.compute_output_shape(output_shape)
        output_shape = self.post.compute_output_shape(output_shape)
        return output_shape


@tf.keras.utils.register_keras_serializable(package=__name__)
class RowConv1D(BaseLayer):
    def __init__(
        self,
        future_context=2,
        activation="clipped_relu",
        regularizer=None,
        initializer=None,
        padding="causal",
        dilation_rate = (1,),
        depth_multiplier=1,
        data_format="channels_last",
        name="rowconv",
        **kwargs,
    ):
        super(RowConv1D, self).__init__(name=name, **kwargs)
        self.future_context = future_context
        self.conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size=future_context * 2 + 1,
            strides=1,
            depth_multiplier=depth_multiplier,
            dilation_rate=dilation_rate,
            data_format=data_format,
            use_bias=False,
            depthwise_regularizer=regularizer,
            depthwise_initializer=initializer,
            bias_regularizer=regularizer,
            bias_initializer=initializer,
            name="rowconv",
            dtype=self.dtype
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="batchnorm",
            gamma_regularizer=regularizer,
            beta_regularizer=regularizer,
            dtype=self.dtype
        )
        self.padding = padding
        self.data_format = data_format
        self.depth_multiplier = depth_multiplier
        self.activation = layer_util.get_activation(name=activation)

    def _compute_causal_padding(self):
        batch_pad = [[0, 0]]
        channel_pad = [[0, 0]]
        height_pad = [[self.conv.dilation_rate[0] * (self.conv.kernel_size[0] - 1), 0]]
        if self.data_format == "channels_last":
            return batch_pad + height_pad + channel_pad
        return batch_pad + channel_pad + height_pad

    def call(self, inputs, training=False):
        padding = self.padding;
        if self.padding == "causal":
            outputs = tf.pad(inputs, self._compute_causal_padding())
            padding = "valid"
        outputs = self.conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs, training=training)
        return outputs

    def compute_output_shape(self, input_shape):
        # output_shape = self.conv.compute_output_shape(input_shape)
        if self.data_format == "channels_first":
            input_dim = input_shape[2]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            input_dim = input_shape[1]
            out_filters = input_shape[2] * self.depth_multiplier

        input_dim = math_util.conv_output_length(
            input_dim,
            self.conv.kernel_size[0],
            padding=self.padding,
            stride=self.conv.strides[0],
            dilation=self.conv.dilation_rate[0],
        )
        if self.data_format == "channels_first":
            output_shape = [input_shape[0], out_filters, input_dim]
        elif self.data_format == "channels_last":
            output_shape = [input_shape[0], input_dim, out_filters]
        else:
            raise ValueError("data_format must be either channels_first or channels_last")
        output_shape = self.bn.compute_output_shape(output_shape)
        return output_shape
    
    def get_config(self):
        config = super(RowConv1D, self).get_config()
        config.update({
            "future_context": self.future_context,
            "activation": self.activation,
            "regularizer": self.conv.depthwise_regularizer,
            "initializer": self.conv.depthwise_initializer,
        })
        return config


@tf.keras.utils.register_keras_serializable(package=__name__)
class RnnBlock(BaseLayer):
    def __init__(
        self,
        rnn_type:str = "lstm",
        units:int = 1024,
        bidirectional:bool = True,
        unroll:bool = False,
        rowconv:int = 0,
        rowconv_activation:str = "clipped_relu",
        dropout:float = 0.1,
        kernel_regularizer:str = None,
        bias_regularizer:str = None,
        initializer:str = None,
        name:str = "rnn_block",
        **kwargs,
    ):
        super(RnnBlock, self).__init__(name=name, **kwargs)
        self.rnn = layer_util.get_rnn(rnn_type)(
            units=units,
            dropout=dropout,
            unroll=unroll,
            return_sequences=True,
            return_state=True,
            use_bias=True,
            name=rnn_type,
            zero_output_for_mask=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            dtype=self.dtype,
        )
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn, name=f"bidirectional_{rnn_type}", dtype=self.dtype)
        self.bn = tf.keras.layers.BatchNormalization(
            name="batch_norm", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.rowconv = None
        if not bidirectional and rowconv > 0:
            self.rowconv = RowConv1D(
                future_context=rowconv,
                name="rowconv",
                regularizer=kernel_regularizer,
                initializer=initializer,
                activation=rowconv_activation,
                dtype=self.dtype,
                padding="causal",
            )

    def get_initial_state(self, batch_size):
        if self.bidirectional:
            states = self.rnn.forward_layer.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype))
            states += self.rnn.backward_layer.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype))
        else:
            states = self.rnn.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype))
        return states

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs, *_ = self.rnn(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        if self.rowconv is not None:
            outputs = self.rowconv(outputs, training=training)
        return outputs, outputs_length

    def call_next(self, inputs, previous_encoder_states):
        with tf.name_scope(f"call_next"):
            outputs, outputs_length = inputs
            outputs, *_states = self.rnn(outputs, training=False, initial_state=tf.unstack(previous_encoder_states, axis=0))
            outputs = self.bn(outputs, training=False)
            if self.rowconv is not None:
                outputs = self.rowconv(outputs, training=False)
            return outputs, outputs_length, tf.stack(_states, axis=0)

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape, *_ = self.rnn.compute_output_shape(output_shape)
        output_shape = self.bn.compute_output_shape(output_shape)
        if self.rowconv is not None:
            output_shape = self.rowconv.compute_output_shape(output_shape)
        return output_shape, output_length_shape
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class RNNModule(BaseLayer):
    def __init__(
        self,
        nlayers: int = 5,
        rnn_type: str = "lstm",
        units: int = 1024,
        bidirectional: bool = True,
        unroll: bool = False,
        rowconv: int = 0,
        rowconv_activation: str = "clipped_relu",
        dropout: float = 0.1,
        kernel_regularizer: str = None,
        bias_regularizer: str = None,
        initializer: str = None,
        name: str = "rnn_module",
        **kwargs,
    ):
        super(RNNModule, self).__init__(name=name, **kwargs)
        self.blocks = [
            RnnBlock(
                rnn_type=rnn_type,
                units=units,
                bidirectional=bidirectional,
                unroll=unroll,
                rowconv=rowconv,
                rowconv_activation=rowconv_activation,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                initializer=initializer,
                name=f"rnn_block_{i}",
                dtype=self.dtype,
            )
            for i in range(nlayers)
        ]
    
    def get_initial_state(self, batch_size):
        states = []
        for block in self.blocks:
            states.append(tf.stack(block.get_initial_state(batch_size=batch_size), axis=0))
        return tf.transpose(tf.stack(states, axis=0), perm=[2, 0, 1, 3])

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs

    def call_next(self, inputs, previous_encoder_states):
        outputs = inputs
        previous_encoder_states = tf.transpose(previous_encoder_states, perm=[1, 2, 0, 3])
        new_states = []
        for i, block in enumerate(self.blocks):
            *outputs, _states = block.call_next(outputs, previous_encoder_states[i])
            new_states.append(_states)
        return outputs, tf.transpose(tf.stack(new_states, axis=0), perm=[2, 0, 1, 3])

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for block in self.blocks:
            output_shape = block.compute_output_shape(output_shape)
        return output_shape


@tf.keras.utils.register_keras_serializable(package=__name__)
class FcBlock(BaseLayer):
    def __init__(
        self,
        units: int = 1024,
        activation: str = "clipped_relu",
        dropout: float = 0.1,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(
            units,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=initializer,
            bias_regularizer=bias_regularizer,
            bias_initializer=initializer,
            name="fc",
            dtype=self.dtype,
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name="bn", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype
        )
        self.act = layer_util.get_activation(activation)
        self.do = tf.keras.layers.Dropout(dropout, name="dropout", dtype=self.dtype)

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        outputs = self.fc(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.act(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.fc.compute_output_shape(output_shape)
        return output_shape, output_length_shape


@tf.keras.utils.register_keras_serializable(package=__name__)
class FCModule(BaseLayer):
    def __init__(
        self,
        nlayers: int = 0,
        units:int = 1024,
        activation:str = "clipped_relu",
        dropout:float = 0.1,
        kernel_regularizer:str = None,
        bias_regularizer:str = None,
        initializer:str = None,
        **kwargs,
    ):
        super(FCModule, self).__init__(**kwargs)
        self.blocks = [
            FcBlock(
                units=units,
                activation=activation,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                initializer=initializer,
                name=f"block_{i}",
                dtype=self.dtype,
            )
            for i in range(nlayers)
        ]

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for block in self.blocks:
            output_shape = block.compute_output_shape(output_shape)
        return output_shape


@tf.keras.utils.register_keras_serializable(package=__name__)
class DeepSpeech2Encoder(tf.keras.Model):
    def __init__(
        self,
        conv_type:str = "conv2d",
        conv_kernels:list = [[11, 41], [11, 21], [11, 21]],
        conv_strides:list = [[2, 2], [1, 2], [1, 2]],
        conv_filters:list = [32, 32, 96],
        conv_padding:str = "same",
        conv_activation:str = "clipped_relu",
        conv_dropout:float = 0.1,
        conv_initializer:str = None,
        rnn_type:str = "lstm",
        rnn_units:int = 1024,
        rnn_nlayers:int = 5,
        rnn_bidirectional:bool = True,
        rnn_unroll:bool = False,
        rnn_rowconv:int = 0,
        rnn_rowconv_activation:str = "clipped_relu",
        rnn_dropout:float = 0.1,
        rnn_initializer:str = None,
        fc_nlayers:int = 0,
        fc_units:int = 1024,
        fc_activation:str = "clipped_relu",
        fc_dropout:float = 0.1,
        fc_initializer:str = None,
        kernel_regularizer:str = None,
        bias_regularizer:str = None,
        initializer:str = None,
        **kwargs,
    ):
        super(DeepSpeech2Encoder, self).__init__(**kwargs)
        self.conv_module = ConvModule(
            conv_type=conv_type,
            kernels=conv_kernels,
            strides=conv_strides,
            filters=conv_filters,
            padding=conv_padding,
            activation=conv_activation,
            dropout=conv_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            initializer=conv_initializer or initializer,
            name="conv_module",
            dtype=self.dtype,
        )

        self.rnn_module = RNNModule(
            nlayers=rnn_nlayers,
            rnn_type=rnn_type,
            units=rnn_units,
            bidirectional=rnn_bidirectional,
            unroll=rnn_unroll,
            rowconv=rnn_rowconv,
            rowconv_activation=rnn_rowconv_activation,
            dropout=rnn_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            initializer=rnn_initializer or initializer,
            name="rnn_module",
            dtype=self.dtype,
        )

        self.fc_module = FCModule(
            nlayers=fc_nlayers,
            units=fc_units,
            activation=fc_activation,
            dropout=fc_dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            initializer=fc_initializer or initializer,
            name="fc_module",
            dtype=self.dtype,
        )
        self.time_reduction_factor = self.conv_module.time_reduction_factor

    def get_initial_state(self, batch_size):
        return self.rnn_module.get_initial_state(batch_size)

    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.conv_module(outputs, training=training)
        outputs = self.rnn_module(outputs, training=training)
        outputs = self.fc_module(outputs, training=training)
        return outputs

    def compute_mask(self, inputs, mask=None):
        outputs = inputs
        mask = self.conv_module.compute_mask(outputs, mask=mask)
        return mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape = self.conv_module.compute_output_shape(output_shape)
        output_shape = self.rnn_module.compute_output_shape(output_shape)
        output_shape = self.fc_module.compute_output_shape(output_shape)
        return output_shape

    

