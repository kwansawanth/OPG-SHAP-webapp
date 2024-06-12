from tensorflow.keras.initializers import Initializer
import tensorflow as tf

class ConvKernalInitializer(Initializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape, dtype=None):
        fan_out = shape[-1]
        fan_in = shape[-2] * shape[0] * shape[1]
        scale = 1.0 / max(1., (fan_in + fan_out) / 2.0)
        limit = tf.sqrt(3.0 * scale)
        return tf.random.uniform(shape, -limit, limit, dtype=dtype)

    def get_config(self):
        return {}
