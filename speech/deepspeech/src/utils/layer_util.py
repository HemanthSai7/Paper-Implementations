import tensorflow as tf
from src.models.activations import ClippedReLU

__all__ = [
    "get_activation",
    "get_conv",
    "get_regularizer",
    "get_rnn",
]

def get_rnn(
    rnn_type: str,
):
    assert rnn_type in ["lstm", "gru", "rnn"]
    if rnn_type == "lstm":
        return tf.keras.layers.LSTM
    if rnn_type == "gru":
        return tf.keras.layers.GRU
    return tf.keras.layers.SimpleRNN

def get_activation(name: str):
    if name == "clipped_relu":
        return ClippedReLU
    elif name == "relu":
        return tf.keras.layers.Activation("relu")
    else:
        raise ValueError(f"Activation {name} not found. Can only be 'relu' or 'clipped_relu'.")

def get_conv(
    conv_type: str,
):
    assert conv_type in ["conv1d", "conv2d"]
    if conv_type == "conv1d":
        return tf.keras.layers.Conv1D
    return tf.keras.layers.Conv2D

def get_regularizer(
    regularizer_config: dict
):
    if regularizer_config["class_name"] == "l1":
        return tf.keras._configs.l1(regularizer_config["value"])
    if regularizer_config["class_name"] == "l2":
        return tf.keras.regularizers.l2(regularizer_config["value"])
    return None