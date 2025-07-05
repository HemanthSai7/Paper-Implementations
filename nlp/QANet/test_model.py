import tensorflow as tf
from src.qanet import QANet
from src.utils.file_util import _readGloveFile, build_embedding_matrix

GLOVE_PATH = 'assets/glove.txt'
EMBEDDING_DIM = 300
wordToIndex, indexToWord, wordToGlove = _readGloveFile(GLOVE_PATH)
embedding_matrix = build_embedding_matrix(wordToIndex, wordToGlove, EMBEDDING_DIM)

word_vocab_size = len(wordToIndex) + 1  # +1 for mask token
print(word_vocab_size)
char_vocab_size = 100
word_embedding_dim = EMBEDDING_DIM
char_embedding_dim = 200
num_filters = 128
kernel_size = 7
num_heads = 8
ffn_dim = 128
num_encoder_blocks = 1
num_model_blocks = 3

batch_size = 2
context_len = 32
query_len = 16
word_len = 10

model = QANet(
    # tokenizer_args
    word_vocab_size=word_vocab_size,
    char_vocab_size=char_vocab_size,
    word_embedding_dim=word_embedding_dim,
    char_embedding_dim=char_embedding_dim,
    num_filters=num_filters,
    kernel_size=kernel_size,
    num_heads=num_heads,
    ffn_dim=ffn_dim,
    num_encoder_blocks=num_encoder_blocks,
    num_model_blocks=num_model_blocks,
    dropout_rate=0.1,
    pretrained_embeddings=embedding_matrix
)

context_word_tokens = tf.zeros((batch_size, context_len), dtype=tf.int32)
context_char_tokens = tf.zeros((batch_size, context_len, word_len), dtype=tf.int32)
query_word_tokens = tf.zeros((batch_size, query_len), dtype=tf.int32)
query_char_tokens = tf.zeros((batch_size, query_len, word_len), dtype=tf.int32)

outputs = model(
    {
        "context_word_ids": context_word_tokens,
        "context_char_ids": context_char_tokens,
        "query_word_ids": query_word_tokens,
        "query_char_ids": query_char_tokens,
    },
    training=False
)
model.summary()

# prob_start, prob_end = outputs

