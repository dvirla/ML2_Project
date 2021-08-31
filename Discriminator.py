import tensorflow as tf
from DenseEnconder import denseenconder


class discriminator(tf.keras.Model):
    def __init__(self, embedding_dim, output_dim, vocab_size):
        super(discriminator, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.output_dim, return_sequences=True, return_state=True)

        self.dense_encoder = denseenconder(self.embedding_dim)
        self.layer1 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, captions, features):
        # features = self.dense_encoder(features)
        captions = self.embedding(captions)

        captions = tf.concat([tf.expand_dims(features, 1), captions], axis=-1)

        h_seq, h_state, final_carry_state = self.lstm(captions)  # TODO: check the thing with last_hidden prediction
        pred = tf.nn.sigmoid(self.layer1(h_state))

        return pred
