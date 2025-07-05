import os
import numpy as np

def _readGloveFile(gloveFile: str | os.PathLike) -> tuple[dict[str, int], dict[int, str], dict[str, np.ndarray]]:
    with open(gloveFile, 'r') as f:
        wordToGlove = {}
        wordToIndex = {}
        indexToWord = {}
        for line in f:
            record = line.strip().split()
            token = record[0]
            wordToGlove[token] = np.array(record[1:], dtype=np.float64)
        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras
            wordToIndex[tok] = kerasIdx
            indexToWord[kerasIdx] = tok
    return wordToIndex, indexToWord, wordToGlove

def build_embedding_matrix(wordToIndex, wordToGlove, embedding_dim=300):
    embedding_matrix = np.random.randn(len(wordToIndex)+1, embedding_dim).astype(np.float32)
    for word, idx in wordToIndex.items():
        vec = wordToGlove.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec
    return embedding_matrix
