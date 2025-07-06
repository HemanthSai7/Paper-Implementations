import tensorflow as tf


class GLU(tf.keras.layers.Activation):
    def __init__(
            self,
            axis = -1,
            name = "glu_activation",
            **kwargs,
    ):
        """Gated Linear Unit activation function
        
        Args:
            axis: int: axis to split the input tensor
            name: str: name of the layer

        Equation:
            GLU(x) = x * sigmoid(x)
        """
        super(GLU, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(
            self,
            inputs,
            **kwargs,
    ):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)
    
    def get_config(self):
        config = super(GLU, self).get_config()
        config.update({
            "axis": self.axis,
        })
        return config