import tensorflow as tf
import numpy as np
from qanet_tf.qanet import QANet
from qanet_tf.loss import squad_loss
from qanet_tf.preprocessing_pipeline import preprocess_data
from qanet_tf.squad_loader import download_squad, load_and_flatten_squad

# --- Hyperparameters ---
BATCH_SIZE = 16
EPOCHS = 1  # Set to higher for real training
MAX_CONTEXT_LEN = 400
MAX_QUERY_LEN = 50
MAX_WORD_LEN = 16
EMBEDDING_DIM = 300

# --- Data Preparation ---
print("Downloading and preprocessing SQuAD v1.1...")
squad_path = download_squad(version="v1.1")
examples = load_and_flatten_squad(squad_path, version="v1.1")
data = preprocess_data(
    examples,
    glove_path="assets/glove.txt",
    embedding_dim=EMBEDDING_DIM,
    stop_words=["an", "a", "the"],
    allowed_delimiters=[",", ".", "'", "?", "!"]
)

# --- Model ---
model = QANet(
    word_vocab_size=data["embedding_matrix"].shape[0],
    char_vocab_size=len(data["char_vocab"]) + 1,
    word_embedding_dim=EMBEDDING_DIM,
    char_embedding_dim=64,
    num_filters=128,
    kernel_size=7,
    num_heads=8,
    ffn_dim=128,
    num_encoder_blocks=1,
    num_model_blocks=3,
    dropout_rate=0.1,
    pretrained_embeddings=data["embedding_matrix"]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# --- tf.data.Dataset Preparation ---
def make_dataset(data, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((
        {
            "context_word_ids": data["context_word_ids"],
            "context_char_ids": data["context_char_ids"],
            "query_word_ids": data["query_word_ids"],
            "query_char_ids": data["query_char_ids"]
        },
        (
            data["start_pos"],
            data["end_pos"]
        )
    ))
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

dataset = make_dataset(data, BATCH_SIZE)

# --- Custom loss for model.compile ---
def custom_squad_loss(y_true, y_pred):
    # y_true: (batch, 2), y_pred: tuple (prob_start, prob_end)
    y_true_start = y_true[:, 0]
    y_true_end = y_true[:, 1]
    y_pred_start, y_pred_end = y_pred
    return squad_loss(
        y_true_start,
        y_true_end,
        y_pred_start,
        y_pred_end
    )

# --- Model Compile ---
model.compile(
    optimizer=optimizer,
    loss=custom_squad_loss
)

# --- Model Fit ---
model.fit(
    dataset,
    epochs=EPOCHS
)
