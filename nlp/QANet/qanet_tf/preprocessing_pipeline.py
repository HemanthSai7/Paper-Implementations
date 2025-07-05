"""
Full preprocessing pipeline for QANet with GloVe and custom tokenizer.
- Loads GloVe
- Builds word and char vocabularies
- Tokenizes and encodes SQuAD examples
- Pads and batches data for model input
"""
import json
import numpy as np
from qanet_tf.utils import _readGloveFile, build_embedding_matrix
from qanet_tf.tokenizer import Tokenizer
from qanet_tf.squad_preprocessing import build_char_vocab, preprocess_squad_examples

# Main pipeline function
def preprocess_data(
    examples,
    glove_path="assets/glove.txt",
    embedding_dim=300,
    stop_words=None,
    allowed_delimiters=None,
    max_context_len=400,
    max_query_len=50,
    max_word_len=16
):
    # 1. Load GloVe and build word vocab/embedding
    wordToIndex, indexToWord, wordToGlove = _readGloveFile(glove_path)
    embedding_matrix = build_embedding_matrix(wordToIndex, wordToGlove, embedding_dim)

    # 2. Initialize tokenizer
    tokenizer = Tokenizer(stop_words=stop_words, allowed_delimiters=allowed_delimiters)

    # 3. Build char vocab from all tokens
    all_tokens = [tokenizer.tokenize(ex['context']) + tokenizer.tokenize(ex['question']) for ex in examples]
    char_vocab = build_char_vocab(all_tokens)

    # 4. Preprocess examples (tokenize, index, pad)
    ctx_wids, ctx_cids, q_wids, q_cids, start_pos, end_pos, is_impossible_mask = preprocess_squad_examples(
        examples, wordToIndex, char_vocab, tokenizer,
        max_context_len=max_context_len,
        max_query_len=max_query_len,
        max_word_len=max_word_len
    )

    return {
        "context_word_ids": ctx_wids,
        "context_char_ids": ctx_cids,
        "query_word_ids": q_wids,
        "query_char_ids": q_cids,
        "embedding_matrix": embedding_matrix,
        "word_vocab": wordToIndex,
        "char_vocab": char_vocab,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "is_impossible_mask": is_impossible_mask
    }

# Example usage (for demo/testing)
if __name__ == "__main__":
    # Dummy SQuAD-like data
    examples = [
        {"context": "The quick brown fox jumps over the lazy dog.", "question": "What does the fox do?"},
        {"context": "TensorFlow is an open source library.", "question": "What is TensorFlow?"}
    ]
    data = preprocess_data(
        examples,
        glove_path="assets/glove.txt",
        embedding_dim=300,
        stop_words=["an", "a", "the"],
        allowed_delimiters=[",", ".", "'", "?", "!"]
    )
    tf.get_logger().info(f"context_word_ids shape: {data['context_word_ids'].shape}")
    tf.get_logger().info(f"context_char_ids shape: {data['context_char_ids'].shape}")
    tf.get_logger().info(f"query_word_ids shape: {data['query_word_ids'].shape}")
    tf.get_logger().info(f"query_char_ids shape: {data['query_char_ids'].shape}")
    tf.get_logger().info(f"embedding_matrix shape: {data['embedding_matrix'].shape}")
    tf.get_logger().info(f"word_vocab size: {len(data['word_vocab'])}")
    tf.get_logger().info(f"char_vocab size: {len(data['char_vocab'])}")
