import tensorflow as tf

def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def shape_list_per_replica(x, per_replica_batch_size):
    print(x.shape)
    shapes = list(x.shape)
    shapes[0] = int(per_replica_batch_size)
    return shapes