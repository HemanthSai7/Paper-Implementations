import tensorflow as tf
from qanet import QANet
from utils import _readGloveFile, build_embedding_matrix

# Load GloVe embeddings
GLOVE_PATH = '/home/hemanth/GIT_Projects/QANet/assets/glove.txt'
EMBEDDING_DIM = 300
wordToIndex, indexToWord, wordToGlove = _readGloveFile(GLOVE_PATH)
embedding_matrix = build_embedding_matrix(wordToIndex, wordToGlove, EMBEDDING_DIM)

# Model hyperparameters
word_vocab_size = len(wordToIndex) + 1  # +1 for mask token
char_vocab_size = 100  # Set as needed
word_embedding_dim = EMBEDDING_DIM
char_embedding_dim = 64
num_filters = 128
kernel_size = 7
num_heads = 8
ffn_dim = 128
num_encoder_blocks = 1
num_model_blocks = 3

# Dummy input shapes
batch_size = 2
context_len = 32
query_len = 16
word_len = 10

# Instantiate model with GloVe embeddings
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

# Build model by calling on dummy data
context_word_ids = tf.zeros((batch_size, context_len), dtype=tf.int32)
context_char_ids = tf.zeros((batch_size, context_len, word_len), dtype=tf.int32)
query_word_ids = tf.zeros((batch_size, query_len), dtype=tf.int32)
query_char_ids = tf.zeros((batch_size, query_len, word_len), dtype=tf.int32)

outputs = model(
    {
        "context_word_ids": context_word_ids,
        "context_char_ids": context_char_ids,
        "query_word_ids": query_word_ids,
        "query_char_ids": query_char_ids,
    },
    training=False
)

# Print output shapes
prob_start, prob_end = outputs
tf.get_logger().info(f"prob_start shape: {prob_start.shape}")
tf.get_logger().info(f"prob_end shape: {prob_end.shape}")

# Print model summary
model.summary()
