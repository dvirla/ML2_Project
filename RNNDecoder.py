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
        # TODO: check option to ommit fc1
        self.layer2 = tf.keras.layers.Dense(vocab_size)
        self.attention = attentionmodel(self.dim)

    @tf.function
    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the LSTM
        h_seq, state, final_carry_state = self.lstm(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.layer1(h_seq)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.layer2(x)

        # TODO: check option to remove attention_weights
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.dim))
