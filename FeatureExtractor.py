import tensorflow as tf


class featureextractor:
    def __init__(self):
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output  # Truncating classifier's layer
        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    @staticmethod
    def load_image(image_path):
        """
        :return: image after preprocess of pre-trained inception_v3 and the path including .jpg
        """
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))  # required for pre-trained model
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

