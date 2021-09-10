import tensorflow as tf


class attentionmodel(tf.keras.Model):
    def __init__(self, dim):
        super(attentionmodel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(dim)
        self.layer2 = tf.keras.layers.Dense(dim)
        self.pred_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, features, hidden):
        """
        :param features: Features of a given batch/image after passing through DenseEncoder
        :param hidden: The last hidden state produced by the lstm layer of RNN Decoder
        :return: context vector = the features multiplied by the attention weights
        """
        # Increasing the dimension by one could help the fc layer choose better features of the hidden state
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # One attention version is extracting vital features both from image features and hidden state using fc layers
        # and then using tanh activation function
        attention_hidden_layer = (tf.nn.tanh(self.layer1(features) + self.layer2(hidden_with_time_axis)))

        # The prediction fc layer outputs an un-normalized score for each image feature
        score = self.pred_layer(attention_hidden_layer)

        # using softmax over the scores of each image feature we get normalized weights for each feature
        attention_weights = tf.nn.softmax(score, axis=1)

        # Multiplying the image features by the attention weights allow amplification of important features
        context_vector = attention_weights * features

        # Using reduce_sum to get one context vector for each image in the batch
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector