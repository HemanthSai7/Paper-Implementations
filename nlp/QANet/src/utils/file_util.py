import os

import numpy as np

def _readGloveFile(gloveFile: str | os.PathLike) -> tuple[dict[str, int], dict[int, str], dict[str, np.ndarray]]:
    with open(gloveFile, 'r') as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token 
        # 123 0.12, 0.32, 0.33, 0.45, ...
        for line in f:
            record = line.strip().split()
            token = record[0] # take the token (word) from the text line
            wordToGlove[token] = np.array(record[1:], dtype=np.float64) # associate the Glove embedding vector to a that token (word)

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras (see above)
            wordToIndex[tok] = kerasIdx # associate an index to a token (word)
            indexToWord[kerasIdx] = tok # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove

def build_embedding_matrix(wordToIndex, wordToGlove, embedding_dim=300):
    embedding_matrix = np.random.randn(len(wordToIndex)+1, embedding_dim).astype(np.float32)
    for word, idx in wordToIndex.items():
        vec = wordToGlove.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec
    return embedding_matrix