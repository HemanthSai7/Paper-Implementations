import tensorflow as tf
from src.qanet import QANet
from src.utils.file_util import _readGloveFile
import numpy as np
import tensorflow.keras.utils as ku

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
        self.wordToIndex, self.indexToWord, self.wordToGlove = _readGloveFile("assets/glove.txt")
        
        self.wordToIndex.update({"<UNK>": len(self.wordToIndex) + 1})
        self.indexToWord.update({len(self.indexToWord) + 1: "<UNK>"})
        self.wordToGlove["<UNK>"] = np.random.rand(300)
        self.max_len = max_len
        
        self.layer = createPretrainedEmbeddingLayer(self.wordToGlove, self.wordToIndex, False)

    def call(self, inputs: list[list[str]]):
        inputs = [[self.wordToIndex.get(word, self.wordToIndex["<UNK>"]) for word in input] for input in inputs]
        inputs = ku.pad_sequences(inputs, padding="post", truncating="post", maxlen=self.max_len)
        return self.layer(inputs=inputs)
    

emb = EmbeddingLayer(max_len=150)

print(emb.wordToIndex[:10])
embedding_layer = emb(inputs=["The quick brown fox, jumps over the lazy dog.", "hi hello"])

model = QANet(
    tokenizer_args={
        "stop_words": ["an", "a", "the"],
        "allowed_delimiters": [",", ".", "'", "?", "!"]
    },
    max_len=150,
    is_serving=False,
    embedding_matrix=embedding_layer,
    vocab_size=len(emb.wordToIndex),
)

res = model(inputs=["The quick brown fox, jumps over the lazy dog.", "hi hello"])
print(res[:10])

# print(model(inputs=["The quick brown fox, jumps over the lazy dog.", "hi hello"]))