import tensorflow as tf
import csv
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 60
BUFFER_SIZE = 1000

random.seed = 42  # Fixing randomness


def parse_images_captions_file(path):
    """
    Reading the image-captions file
    :return: a dictionary {image_name: list of captions}
    """
    img_to_captions_dict = defaultdict(list)
    with open(path) as f:
        csvreader = csv.reader(f, delimiter='\t')
        for row in csvreader:
            img_caption, caption = row
            img = img_caption.split('#')[0]
            img_to_captions_dict[img].append(caption)
    return img_to_captions_dict


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def prepare_images_features():
    images_dir = "Flicker8k_Dataset/"
    tokens_dir = "Flickr8k_text/Flickr8k.token.txt"
    img_to_captions_dict = parse_images_captions_file(tokens_dir)
    # Splitting the image list into train, test sets
    images_list = list(img_to_captions_dict.keys())
    random.shuffle(images_list)
    for p in images_list:
        if len(p.split('.')) > 2:
            print(p)
            exit(0)
    train_images, test_images = images_list[:6000], images_list[6000:]

    # Prepare model for feature extraction
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output  # Truncating classifier's layer
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Mapping images to their paths
    training_image_paths = [f'{images_dir}{name}' for name in train_images]
    image_dataset = tf.data.Dataset.from_tensor_slices(training_image_paths)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

    # Extracting features for each batch, reshaping
    counter = 0
    for img, path in tqdm(image_dataset):
        try:
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = (p.numpy().decode("utf-8")).split('.')[0]
                np.save(path_of_feature, bf.numpy())

        except:
            counter += 1
            print(f'{img} in {path}')
            continue

    return train_images, test_images



if __name__ == "__main__":
    prepare_images_features()
