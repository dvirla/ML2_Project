import tensorflow as tf
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import os

random.seed = 42  # Fixing randomness


class imagefeaturesloader:
    def __init__(self, images_list, images_dirs, feature_extractor):
        self.images_list = images_list
        self.images_dirs = images_dirs
        self.image_dataset = None  # Place holder
        self.images_features_dict = {}  # Place holder
        self.feature_extractor = feature_extractor

    def load_images(self):
        # Splitting the image list into train, test sets
        # random.shuffle(self.images_list)
        images_paths = []
        images_set = set(os.listdir(self.images_dirs[0])).intersection(set(self.images_list))
        for directory in self.images_dirs:
            for img in images_set:
                images_paths.append(f'{directory}{img}')
            # images_paths += [f'{dir}{name}' for name in self.images_list]
        unique_set = sorted(set(images_paths))

        self.image_dataset = tf.data.Dataset.from_tensor_slices(unique_set)
        self.image_dataset = self.image_dataset.map(
            self.feature_extractor.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
            16)  # Mapping images to their paths

        self.images_features_dict = self.create_features()

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
