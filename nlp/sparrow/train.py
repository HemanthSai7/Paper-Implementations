from model import Sparrow
from layers import (
    Tokenizer
)

import tensorflow as tf
import tensorflow.keras.utils as ku
import numpy as np

def train():
    model = Sparrow(
        tokenizer_args={"stop_words": ["an", "a", "the"], "allowed_delimiters":[",", ".", "'", "?", "!"]},
        n_blocks=1,
        n_heads=2,
        n_dims=128,
        max_len=50,
        is_serving=False
    )
    model.compile(optimizer='adam', metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy())

    data = tf.data.Dataset.from_tensor_slices((["Hello ", "the fox is an "], ["world", "animal"]))

    def _prep_inps(inputs):
        inputs = inputs.numpy()
        if isinstance(inputs, bytes):
            inputs = np.array([inputs])
        batch_size = inputs.shape[0]
        
        inputs = [input.decode("utf-8").strip() for input in inputs]
        inputs = model.tokenizer_layer(inputs=inputs)
        inputs = [[model.embedding_layer.wordToIndex.get(word, model.embedding_layer.wordToIndex["<UNK>"]) for word in input] for input in inputs]
        inputs = ku.pad_sequences(inputs, padding="post", truncating="post", maxlen=model.embedding_layer.max_len)
        inputs = tf.Variable(inputs, dtype=tf.int32, shape=(batch_size, model.embedding_layer.max_len))
        # print(inputs)
        return inputs.read_value()
    
    def _prep_outs(outputs):
        outputs = outputs.numpy()
        if isinstance(outputs, bytes):
            outputs = np.array([outputs])
        batch_size = outputs.shape[0]
        
        outputs = [output.decode("utf-8") for output in outputs]
        outputs = [model.embedding_layer.wordToIndex.get(output, model.embedding_layer.wordToIndex["<UNK>"]) for output in outputs]
        outputs = tf.Variable(outputs, dtype=tf.int32, shape=(batch_size,))
        # print(outputs)
        return outputs.read_value()

    def _transform(inputs, outputs):
        inputs = tf.py_function(_prep_inps, inp=[inputs], Tout=tf.int32)
        outputs = tf.py_function(_prep_outs, inp=[outputs], Tout=tf.int32)

        inputs.set_shape([None, model.embedding_layer.max_len])
        outputs.set_shape([None])


        return inputs, outputs
    
    data = data.batch(1).map(_transform)
    for inputs, outputs in data:
        print(inputs, outputs)
    model.build(input_shape=(None, model.embedding_layer.max_len))
    model.fit(data, epochs=10)
    model.is_serving = True
    print([model.embedding_layer.indexToWord.get(res, model.embedding_layer.wordToIndex["<UNK>"]) for res in tf.argmax(model(inputs=["fox"]), axis=-1).numpy()])

with tf.device("/CPU:0"):    
    train()