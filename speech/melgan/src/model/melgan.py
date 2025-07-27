from src.utils import file_util, data_util
from src.schemas import TrainInput
from src.model.encoder import MelGANGenerator
from src.model.decoder import MelGANDiscriminator

import tensorflow as tf

logger = tf.get_logger()


class MelGAN(tf.keras.Model):
    def __init__(
        self, 
        generator: MelGANGenerator,
        discriminator: MelGANDiscriminator,
        name: str = "melgan",
        **kwargs
    ):
        super(MelGAN, self).__init__(name=name, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss_metric = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        self.discriminator_loss_metric = tf.keras.metrics.Mean(name="discriminator_loss", dtype=tf.float32)
        self._tfasr_metrics = {}
        self._tfasr_metrics["loss"] = self.generator_loss_metric
        self._tfasr_metrics["discriminator_loss"] = self.discriminator_loss_metric

    @property
    def metrics(self):
        return list(self._tfasr_metrics.values())
    
    def save(
        self,
        filepath: str,
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: str = None,
        signatures: dict = None,
        options: tf.saved_model.SaveOptions = None,
        save_traces: bool = True,
    ):
        with file_util.save_file(filepath) as path:
            super(MelGAN, self).save(
                filepath=filepath,
                overwrite=overwrite,
                include_optimizer=include_optimizer,
                save_format=save_format,
                signatures=signatures,
                options=options,
                save_traces=save_traces,
            )

    def save_weights(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: str = None,
        options: tf.saved_model.SaveOptions = None,
    ):
        with file_util.save_file(filepath) as path:
            super(MelGAN, self).save_weights(filepath=path, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(
            self,
            filepath,
            by_name=False,
            skip_mismatch=False,
            options=None,
    ):
        with file_util.read_file(filepath) as path:
            super().load_weights(filepath=path, by_name=by_name, skip_mismatch=skip_mismatch, options=options)
    
    def compile(
        self,
        generator_optimizer,
        discriminator_optimizer,
        generator_loss,
        feature_matching_loss,
        discriminator_loss,
        run_eagerly=None,
        **kwargs,
    ):
        super().compile()

        self.generator_optimizer = tf.keras.optimizers.get(generator_optimizer)
        self.discriminator_optimizer = tf.keras.optimizers.get(discriminator_optimizer)

        self.generator_loss = generator_loss
        self.feature_matching_loss = feature_matching_loss
        self.discriminator_loss = discriminator_loss

        self.run_eagerly = run_eagerly

    # def call(self, inputs, training=False, mask=None):
    #     raise NotImplementedError("The call method is not implemented in the base model.")
    
    # --------------------------------------------- Training and Testing Steps -----------------------------------
    def _train_step(self, data):
        x = data[0]
        y = data[1]["text_targets"]

        with tf.GradientTape() as tape:
            tape.watch(x["audio_inputs"])
            outputs = self(x, training=True)
            tape.watch(outputs)
            y_pred = outputs
            loss = self.compute_loss(x, y, y_pred)
            gradients = tape.gradient(loss, self.trainable_variables)

        self.loss_metric.update_state(loss)
        
        return gradients
    
    def train_step(self, data):
        gradients  = self._train_step(data)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
    
    def _test_step(self, data):
        x = data[0]
        y = data[1]["text_targets"]
        outputs = self(x, training=False)
        y_pred = outputs
        loss = self.compute_loss(x, y, y_pred)

        self.loss_metric.update_state(loss)
        return loss
    
    def test_step(self, data):
        self._test_step(data)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        """Clean predict step that uses recognize for autoregressive generation."""
        ...
    
    # --------------------------------------------- TFLITE -----------------------------------

    def recognize(self, signal: tf.Tensor, model_max_length: int = None):
        raise NotImplementedError("The recognize method is not implemented in the base model.")
    
    def recognize_tflite(
        self,
        signal: tf.Tensor,
        predicted: tf.Tensor,
    ):
        raise NotImplementedError("The recognize_tflite method is not implemented in the base model.")
        
    
    def make_tflite_function(beam_width: int = 0):
        raise NotImplementedError("The make_tflite_function method is not implemented in the base model.")