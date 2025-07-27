from src.model.norm import WeightNormConv1D, WeightNormConv1DTranspose

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package=__name__)
class MelGANDiscriminator(tf.keras.layers.Layer):
    def __init__(
        self, 
        num_layers: int = 4, 
        filters: int = 16, 
        kernel_size: int = 15, 
        stride: int = 4, 
        name: str = "melgan_discriminator",
        **kwargs
    ):
        super(MelGANDiscriminator, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_layers = []
        current_filters = filters

        for i in range(num_layers):
            if i == 0:
                conv = tf.keras.layers.Conv1D(
                    filters=current_filters, kernel_size=kernel_size,
                    strides=stride, padding='same'
                )
            else:
                conv = WeightNormConv1D(
                    filters=current_filters, kernel_size=kernel_size,
                    strides=stride, padding='same'
                )
            self.conv_layers.append(conv)
            current_filters *= 4

        self.final_conv = WeightNormConv1D(
            filters=1, kernel_size=3, strides=1, padding='same'
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, training=None):
        x = inputs
        feature_maps = []

        for conv in self.conv_layers:
            x = conv(x, training=training)
            x = self.leaky_relu(x)
            feature_maps.append(x)

        output = self.final_conv(x, training=training)
        feature_maps.append(output)

        return feature_maps
    
    def get_config(self):
        config = super(MelGANDiscriminator, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride
        })
        return config
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class MelGANMultiScaleDiscriminator(tf.keras.Model):
    """Multi-scale discriminator for MelGAN."""

    def __init__(
        self, 
        num_discriminators: int = 3,
        num_layers: int = 4,
        base_filters: int = 16,
        kernel_size: int = 15,
        stride: int = 4,
        name: str = "melgan_multi_scale_discriminator",
        **kwargs
    ):
        super(MelGANMultiScaleDiscriminator, self).__init__(name=name, **kwargs)
        self.num_discriminators = num_discriminators

        self.discriminators = []
        for i in range(num_discriminators):
            self.discriminators.append(
                MelGANDiscriminator(
                    num_layers=num_layers,
                    filters=base_filters,
                    kernel_size=kernel_size,
                    stride=stride
                )
            )

        self.pooling_layers = []
        for i in range(1, num_discriminators):
            self.pooling_layers.append(
                tf.keras.layers.AveragePooling1D(pool_size=4, strides=2, padding='same')
            )

    def call(self, inputs, training=None):
        outputs = []
        x = inputs

        # First discriminator (original resolution)
        outputs.append(self.discriminators[0](x, training=training))

        # Subsequent discriminators (downsampled)
        for pool, disc in zip(self.pooling_layers, self.discriminators[1:]):
            x = pool(x)
            outputs.append(disc(x, training=training))

        return outputs
    
    def get_config(self):
        config = super(MelGANMultiScaleDiscriminator, self).get_config()
        config.update({
            'num_discriminators': self.num_discriminators
        })
        return config