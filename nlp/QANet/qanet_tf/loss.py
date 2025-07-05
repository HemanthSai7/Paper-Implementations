import tensorflow as tf

def squad_loss(y_true_start, y_true_end, y_pred_start, y_pred_end, is_impossible_mask=None):
    """
    Computes loss for SQuAD v1.1/v2.0:
    - y_true_start, y_true_end: true start/end token indices (batch,)
    - y_pred_start, y_pred_end: predicted probabilities (batch, seq_len)
    - is_impossible_mask: (batch,) 1 for unanswerable, 0 for answerable (optional, for v2.0)
    Returns: scalar loss
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    start_loss = loss_fn(y_true_start, y_pred_start)
    end_loss = loss_fn(y_true_end, y_pred_end)
    if is_impossible_mask is not None:
        # Mask out loss for unanswerable
        answerable_mask = 1 - tf.cast(is_impossible_mask, tf.float32)
        start_loss = start_loss * answerable_mask
        end_loss = end_loss * answerable_mask
        denom = tf.reduce_sum(answerable_mask) + 1e-8
        total_loss = tf.reduce_sum(start_loss + end_loss) / denom
    else:
        total_loss = (start_loss + end_loss) / 2.0
    return total_loss
