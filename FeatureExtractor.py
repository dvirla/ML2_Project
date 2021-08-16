import tensorflow as tf


class featureextractor:
    def __init__(self):
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output  # Truncating classifier's layer
        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
