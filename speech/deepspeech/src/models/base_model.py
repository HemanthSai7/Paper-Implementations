from src.featurizers.tokenizer import CharacterTokenizer
from src.utils import file_util, shape_util, data_util
from src.schema import TrainInput, PredictInput, PredictOutputWithTranscript, PredictOutput

import tensorflow as tf

logger = tf.get_logger()

class BaseModel(tf.keras.Model):
    def __init__(
        self,
        **kwargs,
    ):
        super(BaseModel, self).__init__(**kwargs)

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, tokenizer: CharacterTokenizer):
        self._tokenizer = tokenizer

    def summary(
        self,
        line_length=128,
        expand_nested=True,
        show_trainable=True,
        **kwargs,
    ):
        super().summary(line_length=line_length, expand_nested=expand_nested, show_trainable=show_trainable, **kwargs)

    def save(
        self,
        filepath,
        overwrite=True,
        save_format=None,
        **kwargs,
    ):
        with file_util.save_file(filepath) as path:
            super().save(filepath=path, overwrite=overwrite, save_format=save_format, **kwargs)

    def save_weights(
        self,
        filepath,
        overwrite=True,
        save_format=None,
        options=None,
    ):
        with file_util.save_file(filepath) as path:
            super().save_weights(filepath=path, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(
        self,
        filepath,
        by_name=False,
        skip_mismatch=False,
        options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

    def add_custom_metric(self, metric: tf.keras.metrics.Metric):
        if not hasattr(self, "_custom_asr_metrics"):
            self._custom_asr_metrics = {}
        self._custom_asr_metrics[metric.name] = metric

    def make(
        self,
        input_shape=[None],
        prediction_shape=[None],
        batch_size=None,
        **kwargs,
    ):
        assert batch_size is not None and batch_size > 0, "batch_size must be provided and must be positive."
        features = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        features_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        predictions = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        predictions_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        outputs = self(
            TrainInput(
                inputs=features,
                inputs_length=features_length,
                predictions=predictions,
                predictions_length=predictions_length,
            ),
            training=False
        )
        return outputs

    def compile(
        self,
        loss,
        optimizer,
        run_eagerly=None,
        **kwargs,
    ):
        optimizer = tf.keras.optimizers.get(optimizer)
        super().compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly, **kwargs)

    def call(self, inputs: TrainInput, training=False):
        raise NotImplementedError()
    
    def _validate_and_get_metrics_result(self, logs):
        logs = super()._validate_and_get_metrics_result(logs)
        if "predictions" in logs:
            del logs["predictions"]
        return logs
    
    def _train_step(self, data):
        x = data[0]
        y, _ = data_util.set_length(data[1]["labels"], data[1]["labels_length"])

        with tf.GradientTape() as tape:
            tape.watch(x["inputs"])
            outputs = self(x, training=True)
            tape.watch(outputs["logits"])
            y_pred = outputs["logits"]
            y_pred, _ = data_util.set_length(y_pred, outputs["logits_length"])
            tape.watch(y_pred)
            loss = self.compute_loss(x, y, y_pred)
            gradients = tape.gradient(loss, self.trainable_variables)
        return gradients
    
    def train_step(self, data):
        gradients = self._train_step(data)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        metrics = self.get_metrics_result()
        return metrics
    
    def _test_step(self, data):
        x = data[0]
        y, _ = data_util.set_length(data[1]["labels"], data[1]["labels_length"])

        outputs = self(x, training=False)
        y_pred, _ = data_util.set_length(outputs["logits"], outputs["logits_length"])

        self.compute_loss(x, y, y_pred)

    def test_step(self, data):
        self._test_step(data)
        metrics = self.get_metrics_result()
        return metrics
    
    def predict_step(self, data):
        x, y_true = data
        batch_size, *_ = shape_util.shape_list(x["inputs"])
        inputs = PredictInput(
            inputs=x["inputs"],
            inputs_length=x["inputs_length"],
            previous_tokens=self.get_initial_tokens(batch_size=batch_size),
            previous_encoder_states=self.get_initial_encoder_states(batch_size=batch_size),
            previous_decoder_states=self.get_initial_decoder_states(batch_size=batch_size),
        )
        _tokens = self.recognize(inputs=inputs).tokens
        _beam_tokens = self.recognize_beam(inputs=inputs).tokens
        return {
            "_tokens": _tokens,
            "_beam_tokens": _beam_tokens,
            "_labels": y_true["labels"],
        }
    
    # -------------------------------- INFERENCE FUNCTIONS -------------------------------------

    def get_initial_tokens(self, batch_size=1):
        return tf.ones([batch_size, 1], dtype=tf.int32) * self.blank

    def get_initial_encoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

    def get_initial_decoder_states(self, batch_size=1):
        return tf.zeros([], dtype=self.dtype)

    def recognize(self, inputs: PredictInput, **kwargs) -> PredictOutput:
        """Greedy decoding function that used in self.predict_step"""
        raise NotImplementedError()

    def recognize_beam(self, inputs: PredictInput, beam_width: int = 10, **kwargs) -> PredictOutput:
        """Beam search decoding function that used in self.predict_step"""
        raise NotImplementedError()
    
    # ---------------------------------- TFLITE ---------------------------------- #

    def make_tflite_function(self, batch_size: int = 1, beam_width: int = 0):

        def tflite_func(inputs: PredictInput):
            if beam_width > 0:
                outputs = self.recognize_beam(inputs, beam_width=beam_width)
            else:
                outputs = self.recognize(inputs)
            return PredictOutputWithTranscript(
                transcript=self.tokenizer.detokenize(outputs.tokens),
                tokens=outputs.tokens,
                next_tokens=outputs.next_tokens,
                next_encoder_states=outputs.next_encoder_states,
                next_decoder_states=outputs.next_decoder_states,
            )

        input_signature = PredictInput(
            inputs=tf.TensorSpec([batch_size, None], dtype=tf.float32),
            inputs_length=tf.TensorSpec([batch_size], dtype=tf.int32),
            previous_tokens=tf.TensorSpec.from_tensor(self.get_initial_tokens(batch_size)),
            previous_encoder_states=tf.TensorSpec.from_tensor(self.get_initial_encoder_states(batch_size)),
            previous_decoder_states=tf.TensorSpec.from_tensor(self.get_initial_decoder_states(batch_size)),
        )

        return tf.function(
            tflite_func,
            input_signature=[input_signature],
            jit_compile=True,
            reduce_retracing=True,
            autograph=True,
        )