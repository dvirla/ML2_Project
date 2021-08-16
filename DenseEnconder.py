import tensorflow as tf


class denseenconder(tf.keras.Model):
    # This encoder passes the features through a fully connected layer and allows choosing a subset of them
    def __init__(self, embedding_dim):
        super(denseenconder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.layer1 = tf.keras.layers.Dense(embedding_dim)

    @tf.function
    def call(self, x):
        x = self.layer1(x)
        x = tf.nn.relu(x)
        return x