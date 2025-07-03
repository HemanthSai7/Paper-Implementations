import numpy as np

import tensorflow as tf
import tensorflow.keras.utils as ku


from .utils import _readGloveFile

# Create Pretrained Keras Embedding Layer
def createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, isTrainable=False) -> tf.keras.layers.Layer:
    vocabLen = len(wordToIndex) + 1  # adding 1 to account for masking
    embDim = next(iter(wordToGlove.values())).shape[0]  # works with any glove dimensions (e.g. 50)

    embeddingMatrix = np.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        embeddingMatrix[index, :] = wordToGlove[word] # create embedding: word index to Glove word embedding

    embeddingLayer = tf.keras.layers.Embedding(vocabLen, embDim, weights=[embeddingMatrix], trainable=isTrainable)
    return embeddingLayer


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, name="embedding_layer", max_len: int = 50, **kwargs):
        super(EmbeddingLayer, self).__init__(name=name, **kwargs)
        self.wordToIndex, self.indexToWord, self.wordToGlove = _readGloveFile("assets/glove.6B.300d.txt")
        
        self.wordToIndex.update({"<UNK>": len(self.wordToIndex) + 1})
        self.indexToWord.update({len(self.indexToWord) + 1: "<UNK>"})
        self.wordToGlove["<UNK>"] = np.random.rand(300)
        self.max_len = max_len
        
        self.layer = createPretrainedEmbeddingLayer(self.wordToGlove, self.wordToIndex, False)

    def call(self, inputs: list[list[str]]):
        inputs = [[self.wordToIndex.get(word, self.wordToIndex["<UNK>"]) for word in input] for input in inputs]
        inputs = ku.pad_sequences(inputs, padding="post", truncating="post", maxlen=self.max_len)
        return self.layer(inputs=inputs)


class UnembeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, name="unembedding_layer", **kwargs):
        super(UnembeddingLayer, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
    
    def build(self, input_shape): # be last-token
        self.Wu = self.add_weight(
            name = "unembedding_matrix",
            shape = (input_shape[-1], self.vocab_size),
            initializer = "glorot_uniform",
            trainable = True
        )
    
    def call(self, inputs): # be last-token
        return tf.einsum("be,ev->bv", inputs, self.Wu) # bv