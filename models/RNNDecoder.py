import tensorflow as tf
from AttentionModel import attentionmodel


class rnndecoder(tf.keras.Model):
    def __init__(self, embedding_dim, dim, vocab_size):
        super(rnndecoder, self).__init__()
        self.dim = dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dim, return_sequences=True, return_state=True)
        self.layer1 = tf.keras.layers.Dense(self.dim)
        self.layer2 = tf.keras.layers.Dense(vocab_size)
        self.attention = attentionmodel(self.dim)

    @tf.function
    def call(self, captions, features, hidden):
        """
        :param captions: Batch of generated captions
        :param features: Batch of image features
        :param hidden: Last hidden state from the LSTM layer to be used by the attention model
        :return: New generated captions (after adding another word) and the new last hidden state
        """
        # Using the attention model's call function to produce the current context vector
        context_vector = self.attention(features, hidden)

        # Embedding the current captions
        captions = self.embedding(captions)

        # concatenating the context vector and captions to be forward feed into the LSTM layer
        captions = tf.concat([tf.expand_dims(context_vector, 1), captions], axis=-1)
        h_seq, state, final_carry_state = self.lstm(captions)

        # Using two fc layers to produce new generated words out of the new sequence of hidden states
        captions = self.layer1(h_seq)
        captions = tf.reshape(captions, (-1, captions.shape[2]))
        captions = self.layer2(captions)

        return captions, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.dim))
