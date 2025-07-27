import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class WeightNormConv1D(tf.keras.layers.Layer):
    """Conv1D layer with built-in weight normalization."""

    def __init__(
        self, 
        filters,
        kernel_size, 
        strides=1, 
        padding='same', 
        dilation_rate=1, 
        activation=None, 
        use_bias=True, 
        name="weight_norm_conv1d",
        **kwargs
    ):
        super(WeightNormConv1D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        input_channels = input_shape[-1]

        # Weight normalization parameters
        self.v = self.add_weight(
            name='v', shape=(self.kernel_size, input_channels, self.filters),
            initializer='glorot_uniform', trainable=True
        )
        self.g = self.add_weight(
            name='g', shape=(1, 1, self.filters),
            initializer='ones', trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', shape=(self.filters,),
                initializer='zeros', trainable=True
            )

        super(WeightNormConv1D, self).build(input_shape)

    def call(self, inputs, training=None):
        # Normalize weights
        v_norm = tf.nn.l2_normalize(self.v, axis=[0, 1])
        kernel = self.g * v_norm

        # Convolution
        outputs = tf.nn.conv1d(
            inputs, kernel, stride=self.strides,
            padding=self.padding.upper(), dilations=self.dilation_rate
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
    
    def get_config(self):
        config = super(WeightNormConv1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class WeightNormConv1DTranspose(tf.keras.layers.Layer):
    def __init__(
        self, 
        filters, 
        kernel_size,
        strides=1, 
        padding='same', 
        activation=None, 
        use_bias=True, 
        name="weight_norm_conv1d_transpose",
        **kwargs
    ):
        super(WeightNormConv1DTranspose, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        input_channels = input_shape[-1]

        # Weight normalization parameters
        self.v = self.add_weight(
            name='v', shape=(self.kernel_size, self.filters, input_channels),
            initializer='glorot_uniform', trainable=True
        )
        self.g = self.add_weight(
            name='g', shape=(1, self.filters, 1),
            initializer='ones', trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', shape=(self.filters,),
                initializer='zeros', trainable=True
            )

        super(WeightNormConv1DTranspose, self).build(input_shape)

    def call(self, inputs, training=None):
        # Normalize weights
        v_norm = tf.nn.l2_normalize(self.v, axis=[0, 2])
        kernel = self.g * v_norm

        # Calculate output shape
        batch_size = tf.shape(inputs)[0]
        input_length = tf.shape(inputs)[1]
        output_length = input_length * self.strides
        output_shape = tf.stack([batch_size, output_length, self.filters])

        # Transpose convolution
        outputs = tf.nn.conv1d_transpose(
            inputs, kernel, output_shape, strides=self.strides,
            padding=self.padding.upper()
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
    
    def get_config(self):
        config = super(WeightNormConv1DTranspose, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config