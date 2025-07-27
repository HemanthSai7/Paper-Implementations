from src.model.norm import WeightNormConv1D, WeightNormConv1DTranspose
from src.speech_featurizer import SpeechFeaturizer

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class MelGANResidualBlock(tf.keras.layers.Layer):
    """MelGAN residual block with dilated convolutions."""

    def __init__(
        self, 
        filters: int, 
        dilation_rate: int = 1, 
        kernel_size: int = 3, 
        name: str = "melgan_residual_block",
        **kwargs
    ):
        super(MelGANResidualBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size

        self.conv1 = WeightNormConv1D(
            filters=filters, kernel_size=kernel_size,
            dilation_rate=dilation_rate, padding='same'
        )
        self.conv2 = WeightNormConv1D(
            filters=filters, kernel_size=1, padding='same'
        )
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)

    def build(self, input_shape):
        super(MelGANResidualBlock, self).build(input_shape)
        if input_shape[-1] != self.filters:
            self.residual_conv = WeightNormConv1D(
                filters=self.filters, kernel_size=1, padding='same'
            )

    def call(self, inputs, training=None):
        x = self.activation(inputs)
        x = self.conv1(x, training=training)
        x = self.activation(x)
        x = self.conv2(x, training=training)
        return x + inputs
    
    def get_config(self):
        config = super(MelGANResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'dilation_rate': self.dilation_rate,
            'kernel_size': self.kernel_size
        })
        return config
    


@tf.keras.utils.register_keras_serializable(package=__name__)
class MelGANResidualStack(tf.keras.layers.Layer):
    def __init__(
        self, 
        filters: int, 
        dilations: list[int] = [1, 3, 9], 
        name="melgan_residual_stack",
        **kwargs
    ):
        super(MelGANResidualStack, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.dilations = dilations

        self.residual_blocks = []
        for dilation in dilations:
            self.residual_blocks.append(
                MelGANResidualBlock(filters=filters, dilation_rate=dilation)
            )

    def call(self, inputs, training=None):
        x = inputs
        for block in self.residual_blocks:
            x = block(x, training=training)
        return x
    
    def get_config(self):
        config = super(MelGANResidualStack, self).get_config()
        config.update({
            'filters': self.filters,
            'dilations': self.dilations
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class MelGANGenerator(tf.keras.Model):
    """MelGAN Generator network."""

    def __init__(
            self,
            out_channels:int=1,
            upsample_scales:list[int]=[8, 8, 2, 2], 
            filters:int=512,
            dilations:list[int]=[1, 3, 9],
            kernel_size:int=7,
            name="melgan_generator",
            **kwargs
        ):
        super(MelGANGenerator, self).__init__(name=name,**kwargs)

        self.out_channels = out_channels
        self.upsample_scales = upsample_scales
        self.filters = filters
        self.dilations = dilations

        # Initial convolution
        self.input_conv = WeightNormConv1D(
            filters=filters, kernel_size=kernel_size, padding='same'
        )

        # Upsampling layers with residual stacks
        self.upsample_layers = []
        current_filters = filters

        for scale in upsample_scales:
            upsample_layer = WeightNormConv1DTranspose(
                filters=current_filters // 2,
                kernel_size=scale * 2, strides=scale, padding='same'
            )
            residual_stack = MelGANResidualStack(
                filters=current_filters // 2, dilations=dilations
            )
            self.upsample_layers.append((upsample_layer, residual_stack))
            current_filters = current_filters // 2

        # Output layer
        self.output_conv = WeightNormConv1D(
            filters=out_channels, kernel_size=kernel_size,
            padding='same', activation=tf.nn.tanh
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training=None):
        """Generate audio from mel-spectrogram.

        Args:
            inputs: Mel-spectrogram [batch, time, mel_channels]
        Returns:
            Generated audio [batch, time * upsampling_factor, 1]
        """
        x = self.input_conv(inputs, training=training)
        x = self.leaky_relu(x)

        # Upsampling with residual stacks
        for upsample_layer, residual_stack in self.upsample_layers:
            x = upsample_layer(x, training=training)
            x = self.leaky_relu(x)
            x = residual_stack(x, training=training)

        return self.output_conv(x, training=training)
    


@tf.keras.utils.register_keras_serializable(package=__name__)
class MelGANGenerator(tf.keras.Model):
    def __init__(
        self,
        out_channels: int = 1,
        upsample_scales: list[int] = [8, 8, 2, 2],
        filters: int = 512,
        dilations: list[int] = [1, 3, 9],
        kernel_size: int = 7,
        name: str = "melgan_generator",
        **kwargs
    ):
        super(MelGANGenerator, self).__init__(name=name, **kwargs)
        self.out_channels = out_channels
        self.upsample_scales = upsample_scales
        self.filters = filters
        self.dilations = dilations

        # Initial convolution
        self.input_conv = WeightNormConv1D(
            filters=filters, kernel_size=kernel_size, padding='same'
        )

        # Upsampling layers with residual stacks
        self.upsample_layers = []
        current_filters = filters

        for scale in upsample_scales:
            upsample_layer = WeightNormConv1DTranspose(
                filters=current_filters // 2,
                kernel_size=scale * 2, strides=scale, padding='same'
            )
            residual_stack = MelGANResidualStack(
                filters=current_filters // 2, dilations=dilations
            )
            self.upsample_layers.append((upsample_layer, residual_stack))
            current_filters = current_filters // 2

        # Output layer
        self.output_conv = WeightNormConv1D(
            filters=out_channels, kernel_size=kernel_size,
            padding='same', activation=tf.nn.tanh
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training=None):
        """Generate audio from mel-spectrogram.

        Args:
            inputs: Mel-spectrogram [batch, time, mel_channels]
        Returns:
            Generated audio [batch, time * upsampling_factor, 1]
        """
        x = self.input_conv(inputs, training=training)
        x = self.leaky_relu(x)

        # Upsampling with residual stacks
        for upsample_layer, residual_stack in self.upsample_layers:
            x = upsample_layer(x, training=training)
            x = self.leaky_relu(x)
            x = residual_stack(x, training=training)

        return self.output_conv(x, training=training)
    
    def get_config(self):
        config = super(MelGANGenerator, self).get_config()
        config.update({
            'out_channels': self.out_channels,
            'upsample_scales': self.upsample_scales,
            'filters': self.filters,
            'dilations': self.dilations
        })
        return config