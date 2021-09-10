import tensorflow as tf


class discriminator(tf.keras.Model):
    def __init__(self, embedding_dim, output_dim, vocab_size):
        super(discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.output_dim, return_sequences=True, return_state=True)
        self.layer1 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, captions, features):
        """
        :param captions: Fake or real batch of captions
        :param features: Batch of images related to the given captions
        :return: prediction vector of probabilities whether the captions are real (1) or fake (0)
        """
        captions = self.embedding(captions)
        captions = tf.concat([tf.expand_dims(tf.reduce_sum(features, axis=1), 1), captions], axis=1)
        h_seq, h_state, final_carry_state = self.lstm(captions)
        pred = tf.nn.sigmoid(self.layer1(h_state))

        return pred
