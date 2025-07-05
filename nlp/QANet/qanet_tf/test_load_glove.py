from qanet_tf.utils import _readGloveFile, build_embedding_matrix

# Path to GloVe file
GLOVE_PATH = 'assets/glove.txt'
EMBEDDING_DIM = 300  # change if your glove file is different

# Load GloVe
wordToIndex, indexToWord, wordToGlove = _readGloveFile(GLOVE_PATH)
embedding_matrix = build_embedding_matrix(wordToIndex, wordToGlove, EMBEDDING_DIM)

tf.get_logger().info('Loaded GloVe:')
tf.get_logger().info(f'Vocab size: {len(wordToIndex)}')
tf.get_logger().info(f'Embedding matrix shape: {embedding_matrix.shape}')
tf.get_logger().info(f'First word: {indexToWord[1]}')
tf.get_logger().info(f'First embedding: {embedding_matrix[1][:10]}')
