import tensorflow as tf

from src.models.layers import BaseLayer
from src.utils import layer_util
from src.models.base_ctc import CTCModel
from src.models.deepspeech import DeepSpeech2Encoder


__all__ = ["DeepSpeech2"]

@tf.keras.utils.register_keras_serializable(package=__name__)
class DeepSpeech2Decoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        kernel_regularizer=None,
        bias_regularizer=None,
        initializer=None,
        name: str = "deepspeech2_decoder",
        **kwargs,
    ):
        super(DeepSpeech2Decoder, self).__init__(name=name, **kwargs)
        self.vocab = tf.keras.layers.Dense(
            vocab_size,
            kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
            kernel_initializer=tf.keras.initializers.get(initializer),
            bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        logits, logits_length = inputs
        logits = self.vocab(logits, training=training)
        return logits, logits_length
    
    def call_next(self, logits, logits_length, *args, **kwargs):
        outputs, outputs_length = self((logits, logits_length), training=False)
        return outputs, outputs_length

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = self.vocab.compute_output_shape(output_shape)
        return output_shape, output_length_shape
    

@tf.keras.utils.register_keras_serializable(package=__name__)
class DeepSpeech2(CTCModel):
    def __init__(
        self,
        blank: int,
        vocab_size: int,
        conv_type: str = "conv2d",
        conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
        conv_strides: list = [[3, 2], [1, 2], [1, 2]],
        conv_filters: list = [32, 32, 96],
        conv_padding: str = "same",
        conv_activation: str = "relu",
        conv_dropout: float = 0.1,
        rnn_nlayers: int = 5,
        rnn_type: str = "lstm",
        rnn_units: int = 1024,
        rnn_bidirectional: bool = True,
        rnn_unroll: bool = False,
        rnn_rowconv: int = 0,
        rnn_rowconv_activation: str = "relu",
        rnn_dropout: float = 0.1,
        fc_nlayers: int = 0,
        fc_units: int = 1024,
        fc_activation: str = "relu",
        fc_dropout: float = 0.1,
        name: str = "deepspeech2",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_initializer="he_uniform",
        bias_initializer="zeros",
        initializer=None,
        **kwargs,
    ):
        super().__init__(
            blank=blank,
            encoder=DeepSpeech2Encoder(
                conv_type=conv_type,
                conv_kernels=conv_kernels,
                conv_strides=conv_strides,
                conv_filters=conv_filters,
                conv_padding=conv_padding,
                conv_activation=conv_activation,
                conv_dropout=conv_dropout,
                conv_initializer=tf.keras.initializers.get(kernel_initializer),
                rnn_nlayers=rnn_nlayers,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                rnn_bidirectional=rnn_bidirectional,
                rnn_unroll=rnn_unroll,
                rnn_rowconv=rnn_rowconv,
                rnn_rowconv_activation=rnn_rowconv_activation,
                rnn_dropout=rnn_dropout,
                rnn_initializer=tf.keras.initializers.get(kernel_initializer),
                fc_nlayers=fc_nlayers,
                fc_units=fc_units,
                fc_activation=fc_activation,
                fc_dropout=fc_dropout,
                fc_initializer=tf.keras.initializers.get(kernel_initializer),
                kernel_regularizer=tf.keras.regularizers.get(kernel_regularizer),
                bias_regularizer=tf.keras.regularizers.get(bias_regularizer),
                initializer=tf.keras.initializers.get(initializer),
                name="encoder",
            ),
            decoder=DeepSpeech2Decoder(
                vocab_size=vocab_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                initializer=initializer,
                name="decoder",
            ),
            name=name,
            **kwargs,
        )
        self.time_reduction_factor = self.encoder.time_reduction_factor

    def get_initial_encoder_states(self, batch_size=1):
        return self.encoder.get_initial_state(batch_size)

    def get_initial_decoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)
            