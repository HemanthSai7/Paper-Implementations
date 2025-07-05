import tensorflow as tf

def float_feature(
    list_of_floats,
):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def int64_feature(
    list_of_ints,
):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def bytestring_feature(
    list_of_bytestrings,
):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))