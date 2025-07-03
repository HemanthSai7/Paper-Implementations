import tensorflow as tf


__all__ = [
    "MLP"
]

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units: int,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.n_units = n_units
    
    def build(self, input_shape): # bse
        self.W1 = self.add_weight(
            name = "ffn_W1",
            shape = (input_shape[1], input_shape[-1], self.n_units),
            initializer = "glorot_uniform",
            trainable = True
        )

        self.b1 = self.add_weight(
            name = "ffn_b1",
            shape = (input_shape[1], self.n_units,),
            initializer = "zeros",
            trainable = True
        )

        self.W2 = self.add_weight(
            name = "ffn_W2",
            shape = (input_shape[1], self.n_units, input_shape[-1]),
            initializer = "glorot_uniform",
            trainable = True
        )

        self.b2 = self.add_weight(
            name = "ffn_b2",
            shape = (input_shape[1], input_shape[-1],),
            initializer = "zeros",
            trainable = True
        )
    
    def call(self, inputs):
        x = tf.nn.relu(tf.einsum("bse,seu->bsu", inputs, self.W1) + self.b1)
        x = tf.einsum("bsu,sue->bse", x, self.W2) + self.b2
        return x


class MLP(tf.keras.layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()

    def build(self, input_shape): # bse
        self.ffn = FeedForwardNetwork(n_units=input_shape[-1] * 4)
    
    def call(self, inputs): # bse
        x = self.ffn(inputs=inputs)
        return x + inputs # bse
