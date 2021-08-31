import tensorflow as tf
from AttentionModel import attentionmodel


class rnndecoder(tf.keras.Model):
    def __init__(self, embedding_dim, dim, vocab_size):
        super(rnndecoder, self).__init__()
        self.dim = dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # TODO: try with bidirectional
        self.lstm = tf.keras.layers.LSTM(self.dim, return_sequences=True, return_state=True)
        self.layer1 = tf.keras.layers.Dense(self.dim)
        self.layer2 = tf.keras.layers.Dense(vocab_size)
        self.attention = attentionmodel(self.dim)

    @tf.function
    def call(self, captions, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # captions shape after passing through embedding == (batch_size, 1, embedding_dim)
        captions = self.embedding(captions)
        # captions shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        captions = tf.concat([tf.expand_dims(context_vector, 1), captions], axis=-1)

        # passing the concatenated vector to the LSTM
        h_seq, state, final_carry_state = self.lstm(captions)

        # shape == (batch_size, max_length, hidden_size)
        captions = self.layer1(h_seq)

        # captions shape == (batch_size * max_length, hidden_size)
        captions = tf.reshape(captions, (-1, captions.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        captions = self.layer2(captions)

        return captions, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.dim))
