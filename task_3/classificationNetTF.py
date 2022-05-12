import tensorflow as tf
import keras
from keras import layers, activations


class TinyModule(tf.keras.Model):
    def __init__(self):
        # Initialize the necessary components of tf.keras.Model
        super(TinyModule, self).__init__()

        self.dense1 = keras.layers.Dense(3000, activation=tf.nn.leaky_relu)
        self.dropout = keras.layers.Dropout(.7)
        self.dense2 = keras.layers.Dense(1000, activation=tf.nn.leaky_relu)
        self.dense3 = keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x  # Return results of Output Layer
