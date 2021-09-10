import tensorflow as tf


class denseenconder(tf.keras.Model):
    # This encoder passes the features through a fully connected layer and allows choosing a subset of them
    def __init__(self, embedding_dim):
        super(denseenconder, self).__init__()
        self.layer1 = tf.keras.layers.Dense(embedding_dim)

    @tf.function
    def call(self, features):
        """
        :param features: Features of a given batch/image.
        :return: x = The features after passing through a fc layer and then relu function to try and find best features
        """
        features = self.layer1(features)
        x = tf.nn.relu(features)
        return x