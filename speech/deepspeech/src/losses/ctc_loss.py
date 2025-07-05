from src.utils import data_util, math_util

import tensorflow as tf

logger = tf.get_logger()


class CTCLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        blank=0,
        reduction="sum_over_batch_size",
        name=None,
    ):
        super(CTCLoss, self).__init__(reduction=reduction, name=name)
        assert blank==0, "Only blank=0 is supported for now."
        self.blank = blank
        logger.info(f"Using CTC Loss with blank={self.blank}")

    def call(self, y_true, y_pred):
        logits = y_pred
        logit_length = data_util.get_length(logits)
        if logit_length is None:
            logit_length = tf.shape(logits, out_type=tf.int32)[1] * tf.ones(shape=(tf.shape(logits)[0],), dtype=tf.int32)

        labels = y_true
        label_length = data_util.get_length(labels)
        if label_length is None:
            label_length = math_util.count_non_blank(labels, blank=self.blank, axis=1)

        labels = tf.sparse.from_dense(labels)
        return tf.nn.ctc_loss(
            logits=logits,
            logit_length=logit_length,
            labels=labels,
            label_length=label_length,
            logits_time_major=False,
            unique=None,
            blank_index=self.blank,
            name=self.name
        )
    