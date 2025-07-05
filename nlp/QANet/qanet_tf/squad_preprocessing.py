"""
SQuAD preprocessing pipeline for QANet using custom Tokenizer.
- Tokenizes context/questions
- Maps to word and char indices
- Pads/truncates as needed for batching
"""
import json
from typing import List, Dict
import numpy as np
from qanet_tf.utils import _readGloveFile, build_embedding_matrix
from src.tokenizer import Tokenizer

# Utility to build char vocab from dataset
def build_char_vocab(all_tokens: List[List[str]]) -> Dict[str, int]:
    chars = set()
    for tokens in all_tokens:
        for token in tokens:
            chars.update(token)
    char_list = sorted(list(chars))
    char_vocab = {c: i+1 for i, c in enumerate(char_list)}  # 0 is reserved for padding
    char_vocab["[UNK]"] = len(char_vocab) + 1
    return char_vocab

def pad_sequences(seqs, maxlen, pad_value=0):
    arr = np.full((len(seqs), maxlen), pad_value, dtype=np.int32)
    for i, seq in enumerate(seqs):
        arr[i, :min(len(seq), maxlen)] = seq[:maxlen]
    return arr

def pad_char_sequences(seqs, max_sent_len, max_word_len, pad_value=0):
    arr = np.full((len(seqs), max_sent_len, max_word_len), pad_value, dtype=np.int32)
    for i, word_seq in enumerate(seqs):
        for j, char_seq in enumerate(word_seq[:max_sent_len]):
            arr[i, j, :min(len(char_seq), max_word_len)] = char_seq[:max_word_len]
    return arr

def char_to_token_span(context, tokens, answer_start, answer_text):
    """
    Map character-level answer span to token-level span.
    Returns (start_token_idx, end_token_idx) or (None, None) if not found.
    """
    char_to_token = []
    token_idx = 0
    char_idx = 0
    for token in tokens:
        # Find token in context starting from char_idx
        while char_idx < len(context) and context[char_idx].isspace():
            char_idx += 1
        if context[char_idx:char_idx+len(token)] == token:
            for _ in range(len(token)):
                char_to_token.append(token_idx)
                char_idx += 1
            token_idx += 1
        else:
            # Fallback: skip a char and try again
            char_idx += 1
            char_to_token.append(token_idx)
    # Find start and end
    start_char = answer_start
    end_char = answer_start + len(answer_text) - 1
    if start_char < len(char_to_token) and end_char < len(char_to_token):
        return char_to_token[start_char], char_to_token[end_char]
    return None, None

def preprocess_squad_examples(
    examples, word_vocab, char_vocab, tokenizer, max_context_len=400, max_query_len=50, max_word_len=16
):
    context_word_ids, context_char_ids = [], []
    query_word_ids, query_char_ids = [], []
    start_pos, end_pos, is_impossible_mask = [], [], []
    for ex in examples:
        ctx_tokens = tokenizer.tokenize(ex['context'])
        q_tokens = tokenizer.tokenize(ex['question'])
        context_word_ids.append(tokenizer.tokens_to_indices(ctx_tokens, word_vocab))
        context_char_ids.append(tokenizer.tokens_to_char_indices(ctx_tokens, char_vocab, max_word_len))
        query_word_ids.append(tokenizer.tokens_to_indices(q_tokens, word_vocab))
        query_char_ids.append(tokenizer.tokens_to_char_indices(q_tokens, char_vocab, max_word_len))
        # Answer span extraction
        if ex.get('is_impossible', False) or not ex.get('answers'):
            # No answer (v2.0 unanswerable)
            start_pos.append(0)
            end_pos.append(0)
            is_impossible_mask.append(1)
        else:
            # Use first answer for training
            answer_start = ex['answer_starts'][0]
            answer_text = ex['answers'][0]
            start_idx, end_idx = char_to_token_span(ex['context'], ctx_tokens, answer_start, answer_text)
            if start_idx is None or end_idx is None:
                # Fallback: mark as unanswerable
                start_pos.append(0)
                end_pos.append(0)
                is_impossible_mask.append(1)
            else:
                start_pos.append(min(start_idx, max_context_len-1))
                end_pos.append(min(end_idx, max_context_len-1))
                is_impossible_mask.append(0)
    context_word_ids = pad_sequences(context_word_ids, max_context_len)
    context_char_ids = pad_char_sequences(context_char_ids, max_context_len, max_word_len)
    query_word_ids = pad_sequences(query_word_ids, max_query_len)
    query_char_ids = pad_char_sequences(query_char_ids, max_query_len, max_word_len)
    start_pos = np.array(start_pos, dtype=np.int32)
    end_pos = np.array(end_pos, dtype=np.int32)
    is_impossible_mask = np.array(is_impossible_mask, dtype=np.int32)
    return context_word_ids, context_char_ids, query_word_ids, query_char_ids, start_pos, end_pos, is_impossible_mask

# Example usage (for demo/testing)
if __name__ == "__main__":
    # Load GloVe
    GLOVE_PATH = "assets/glove.txt"
    EMBEDDING_DIM = 300
    wordToIndex, indexToWord, wordToGlove = _readGloveFile(GLOVE_PATH)
    embedding_matrix = build_embedding_matrix(wordToIndex, wordToGlove, EMBEDDING_DIM)
    # Dummy SQuAD-like data
    examples = [
        {"context": "The quick brown fox jumps over the lazy dog.", "question": "What does the fox do?"},
        {"context": "TensorFlow is an open source library.", "question": "What is TensorFlow?"}
    ]
    # Tokenizer
    tokenizer = Tokenizer(stop_words=["an", "a", "the"], allowed_delimiters=[",", ".", "'", "?", "!"])
    # Build char vocab from all tokens in dataset
    all_tokens = [tokenizer.tokenize(ex['context']) + tokenizer.tokenize(ex['question']) for ex in examples]
    char_vocab = build_char_vocab(all_tokens)
    # Preprocess
    ctx_wids, ctx_cids, q_wids, q_cids = preprocess_squad_examples(
        examples, wordToIndex, char_vocab, tokenizer)
    print("context_word_ids shape:", ctx_wids.shape)
    print("context_char_ids shape:", ctx_cids.shape)
    print("query_word_ids shape:", q_wids.shape)
    print("query_char_ids shape:", q_cids.shape)
