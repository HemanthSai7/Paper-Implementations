import tensorflow as tf
import numpy as np

from src.tokenizer import TokenizerLayer


def train():
    data = tf.data.Dataset.from_tensor_slices((["Hello ", "the fox is an "], ["world", "animal"]))

    def _prep_inps(inputs):
        inputs = inputs.numpy()
        if isinstance(inputs, bytes):
            inputs = np.array([inputs])
        batch_size = inputs.shape[0]

        inputs = [input.decode("utf-8").strip() for input in inputs]
        inputs = TokenizerLayer(inputs)
        return inputs
    
    def transform(inputs):
        inputs = tf.py_function(_prep_inps, inp=[inputs], Tout=tf.int32)
        inputs.set_shape([None, None])

        return inputs
    
    transform(data)
    

if __name__ == "__main__":
    train()
