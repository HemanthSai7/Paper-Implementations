from src.schema import (
    TrainInput,
    TrainOutput,
    TrainLabel,
    PredictOutput
)
from src.losses.ctc_loss import CTCLoss
from src.models.base_model import BaseModel
from src.utils import layer_util

import tensorflow as tf


class CTCModel(BaseModel):
    def __init__(
        self,
        blank: int,
        encoder: tf.keras.layers.Layer,
        decoder: tf.keras.layers.Layer,
        **kwargs,
    ):
        super(CTCModel, self).__init__(**kwargs)
        self.blank = blank
        self.encoder = encoder
        self.decoder = decoder
        self.time_reduction_factor = 1

    def compile(self, optimizer, loss, output_shapes=None, **kwargs):
        return super().compile(loss, optimizer, **kwargs)
    
    def call(self, inputs, training=False):
        logits, logits_length = self.encoder((inputs["inputs"], inputs["inputs_length"]), training=training)
        logits, logits_length = self.decoder((logits, logits_length), training=training)
        return TrainOutput(
            logits=logits,
            logits_length=logits_length,
        )
    
    def call_next(
        self,
        inputs,
        previous_encoder_states=None,
        previous_decoder_states=None,
    ):
        outputs, outputs_length, next_encoder_states = self.encoder.call_next(inputs,previous_encoder_states,previous_decoder_states)
        outputs, outputs_length, next_decoder_states = self.decoder.call_next(outputs,outputs_length,previous_decoder_states)
        return outputs, outputs_length, next_encoder_states, next_decoder_states
    
    def get_initial_encoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)
    
    def get_initial_decoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

    def recognize(self, inputs, **kwargs):
        with tf.name_scope(f"{self.name}_recognize"):
            outputs, outputs_length, next_encoder_states, next_decoder_states = self.call_next(inputs["input"], inputs["inputs_length"], inputs["previous_encoder_states"], inputs["previous_decoder_states"])
            tokens, _ = tf.nn.ctc_greedy_decoder(
                inputs=tf.transpose(outputs, perm=[1, 0, 2]),
                sequence_length=outputs_length,
                merge_repeated=True,
                blank=self.blank,
            )
            tokens = tf.sparse.to_dense(tokens[0])
            tokens = tf.cast(tokens, dtype=tf.int32)
            return PredictOutput(
                tokens=tokens,
                next_token=None,
                next_encoder_states=next_encoder_states,
                next_decoder_states=next_decoder_states,
            )
        
    def recognize_beam(self, inputs, beam_width: int = 10, **kwargs):
        with tf.name_scope(f"{self.name}_recognize_beam"):
            outputs, outputs_length, next_encoder_states, next_decoder_states = self.call_next(inputs["input"], inputs["inputs_length"], inputs["previous_encoder_states"], inputs["previous_decoder_states"])
            tokens, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf.transpose(outputs, perm=[1, 0, 2]),
                sequence_length=outputs_length,
                beam_width=beam_width,
                merge_repeated=True,
                blank=self.blank,
            )
            tokens = tf.sparse.to_dense(tokens[0])
            tokens = tf.cast(tokens, dtype=tf.int32)
            return PredictOutput(
                tokens=tokens,
                next_token=None,
                next_encoder_states=next_encoder_states,
                next_decoder_states=next_decoder_states,
            )