import tensorflow as tf

__all__ = [
    "TransformerScheduler",
]

@tf.keras.utils.register_keras_serializable(package=__name__)
class TransformerScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dmodel: int, scale: float = 1.0, warmup_steps: int = 4000, max_lr: float = None, min_lr: float = None):
        super().__init__()
        self.dmodel = tf.convert_to_tensor(dmodel, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)
        self.warmup_steps = tf.convert_to_tensor(warmup_steps, dtype=tf.float32)
        self.max_lr = eval(max_lr) if isinstance(max_lr, str) else max_lr
        self.min_lr = eval(min_lr) if isinstance(min_lr, str) else min_lr

    def __call__(self, current_step):
        # lr = (d_model^-0.5) * min(step^-0.5, step*(warm_up^-1.5))
        step = tf.cast(current_step, dtype=tf.float32)
        lr = (self.dmodel**-0.5) * tf.math.minimum(step**-0.5, step * (self.warmup_steps**-1.5))
        lr = self.scale * lr
        if self.max_lr is not None:
            lr = tf.math.minimum(self.max_lr, lr)
        if self.min_lr is not None:
            lr = tf.math.maximum(self.min_lr, lr)
        return lr

    def get_config(self):
        return {
            "dmodel": int(self.dmodel.numpy()),
            "scale": float(self.scale.numpy()),
            "warmup_steps": int(self.warmup_steps.numpy()),
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }