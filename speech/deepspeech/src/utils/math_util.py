from src.utils import shape_util

import math
import tensorflow as tf

def log10(x):
    return tf.math.log(x) / tf.math.log(10.0)

def merge_two_last_dims(x):
    b, _, f, c = shape_util.shape_list(x)
    return tf.reshape(x, shape=[b, -1, f * c])

def get_num_batches(
    nsamples,
    batch_size,
    drop_remainders=True,
):
    if nsamples is None or batch_size is None:
        return None
    if drop_remainders:
        return math.floor(float(nsamples) / float(batch_size))
    return math.ceil(float(nsamples) / float(batch_size))

def get_nsamples(
    duration: float,
    sample_rate: int = 16000,
):
    return math.ceil(float(duration) * sample_rate)

def count_non_blank(
    tensor: tf.Tensor, 
    blank: int = 0, 
    axis=None, 
    dtype=tf.int32, 
    keepdims=False
):
    return tf.reduce_sum(
        tf.where(
            tf.not_equal(tf.cast(tensor, dtype), blank),
            x=tf.ones_like(tensor, dtype=dtype),
            y=tf.zeros_like(tensor, dtype=dtype)
        ),
        axis=axis,
        keepdims=keepdims
    )
    
def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    Args:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full", "causal"
        stride: integer.
        dilation: dilation rate, integer.
    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {"same", "valid", "full", "causal"}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ["same", "causal"]:
        output_length = input_length
    elif padding == "valid":
        output_length = input_length - dilated_filter_size + 1
    elif padding == "full":
        output_length = input_length + dilated_filter_size - 1
    else:
        raise ValueError(f"Invalid padding: {padding}")
    return (output_length + stride - 1) // stride