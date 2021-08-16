import tensorflow as tf
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np

random.seed = 42  # Fixing randomness


class imagefeaturesloader:
    def __init__(self, images_list, images_dir, feature_extractor):
        self.images_list = images_list
        self.images_dir = images_dir
        self.image_dataset = None  # Place holder
        self.images_features_dict = {}  # Place holder
        self.feature_extractor = feature_extractor

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

    def load_images(self):
        # Splitting the image list into train, test sets
        random.shuffle(self.images_list)
        train_images, test_images = self.images_list[:6000], self.images_list[6000:]

        training_image_paths = [f'{name}' for name in train_images]
        unique_set = sorted(set(training_image_paths))

        self.image_dataset = tf.data.Dataset.from_tensor_slices(unique_set)
        self.image_dataset = self.image_dataset.map(
            self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            16)  # Mapping images to their paths

        self.images_features_dict = self.create_features()

        return train_images, test_images

    def create_features(self):
        """
        :return: images_features_dict maps image path to its features extracted from feature_extractor
        """
        images_features_dict = defaultdict(np.array)
        # Extracting features for each batch, reshaping
        for img, path in tqdm(self.image_dataset):
            try:
                batch_features = self.feature_extractor.image_features_extract_model(img)
                batch_features = tf.reshape(batch_features,
                                            (batch_features.shape[0], -1, batch_features.shape[3]))

                for bf, p in zip(batch_features, path):
                    images_features_dict[p.numpy().decode("utf-8")] = bf.numpy()
            except:
                continue
        return images_features_dict
