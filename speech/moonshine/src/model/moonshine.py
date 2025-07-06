from src.model import MoonshineEncoder, MoonshineDecoder
from src.model import BaseModel

import tensorflow as tf

logger = tf.get_logger()


__all__ = ["Moonshine"]


class Moonshine(BaseModel):
    def __init__(
            self,
            vocab_size: int, 
            d_model: int = 288, 
            encoder_num_blocks: int = 6,
            decoder_num_blocks: int = 6,
            fc_factor: int = 4,
            encoder_activation: str = "gelu",
            decoder_activation: str = "swiglu",
            num_heads: int = 8,
            head_size: int = 64,
            dropout: float = 0.0,
            kernel_initializer: str = "glorot_uniform",
            kernel_regularizer: str = tf.keras.regularizers.l2(0.0005),
            bias_regularizer: str = tf.keras.regularizers.l2(0.0005),
            bias_initializer: str = "zeros",
            name: str = "moonshine", 
            **kwargs
        ):
        super(Moonshine, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder = MoonshineEncoder(
            input_dim=d_model,
            num_blocks=encoder_num_blocks,
            fc_factor=fc_factor,
            activation=encoder_activation,
            head_size=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_initializer=bias_initializer,
            name=f"{name}_encoder",
        )
        self.decoder = MoonshineDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            decoder_num_blocks=decoder_num_blocks,
            num_heads=num_heads,
            head_size=head_size,
            fc_factor=fc_factor,
            activation=decoder_activation,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            bias_initializer=bias_initializer,
            name=f"{name}_decoder",
        )

    def call(self, inputs, training=False, mask=None):
        try:
            inputs, inputs_length, predictions, predictions_length = inputs["inputs"], inputs["inputs_length"], inputs["predictions"], inputs["predictions_length"]

            encoder_outputs = self.encoder(inputs=[inputs, inputs_length], training=training, mask=mask)
            logits = self.decoder(inputs=[encoder_outputs, predictions, predictions_length], training=training)
             
            if training:
                return tf.nn.softmax(logits, axis=-1)
            else:
                return tf.argmax(logits, axis=-1)
        except Exception as e:
            import traceback
            logger.error(f"Error in model call: {e}")
            logger.info(traceback.format_exc())
            raise
    
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            inputs_shape = input_shape["inputs"]
            inputs_length_shape = input_shape["inputs_length"]
            predictions_shape = input_shape["predictions"]
            predictions_length_shape = input_shape["predictions_length"]
        else:
            inputs_shape, inputs_length_shape, predictions_shape, predictions_length_shape = input_shape

        encoder_outputs_shape = self.encoder.compute_output_shape([inputs_shape, inputs_length_shape])
        decoder_outputs_shape = self.decoder.compute_output_shape([encoder_outputs_shape, predictions_shape, predictions_length_shape])

        outputs_shape = decoder_outputs_shape
        
        return outputs_shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "encoder_num_blocks": self.encoder.encoder_num_blocks,
            "decoder_num_blocks": self.decoder.decoder_num_blocks,
            "fc_factor": self.encoder.fc_factor,
            "num_heads": self.encoder.num_heads,
            "head_size": self.encoder.head_size,
        })
        return config