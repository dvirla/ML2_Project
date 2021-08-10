import tensorflow as tf
import csv
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

BATCH_SIZE = 60
BUFFER_SIZE = 1000

random.seed = 42  # Fixing randomness
images_dir = "Flicker8k_Dataset/"
tokens_dir = "Flickr8k_text/Flickr8k.token.txt"


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

class prepare_captions():
    def __init__(self, img_to_captions_dict, train_imgs, test_imgs):
        # general variables
        self.img_to_captions_dict = img_to_captions_dict
        self.processed_img_to_captions_dict = defaultdict(list)
        self.padded_captions_idxs_dict = defaultdict(list)
        self.all_captions = []
        self.tokenizer = Tokenizer()
        self.max_caption_len = 0
        self.vocab_size = 0
        self.train_imgs = train_imgs
        self.test_imgs = test_imgs

        # functions
        self.captions_preprocess()
        self.tokenizer_fit()

    def captions_preprocess(self):
        for img, caption_list in self.img_to_captions_dict.items():
            processed_captions = []
            for caption in caption_list:
                # allow only word and numbers
                processed_caption = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
                #  allow only words of more than one letter
                processed_caption_list = [word for word in processed_caption.split(' ') if ((len(word) > 1) and (word.isalpha()))]
                #  add beginning and ending of caption tokens
                processed_caption_list = ['start_cap'] + processed_caption_list + ['end_cap']
                self.max_caption_len = max([self.max_caption_len, len(processed_caption_list)])
                processed_caption = ' '.join(processed_caption_list)
                self.all_captions.append(processed_caption)
                processed_captions.append(processed_caption)
            self.processed_img_to_captions_dict[img] = processed_captions

    def tokenizer_fit(self):
        self.tokenizer.fit_on_texts(self.all_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def caption_padding(self):
        for img, caption_list in self.processed_img_to_captions_dict.items():
            for caption in caption_list:
                caption_idxs = self.tokenizer.texts_to_sequences([caption])[0]
                padded_idxs = pad_sequences([caption_idxs], maxlen=self.max_caption_len, padding='post')[0] #todo: maybe value of the padding from zero?
                self.padded_captions_idxs_dict[img].append(padded_idxs)

    def split_dic_to_train_set(self, directory):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for img in self.train_imgs:
            for padded_idx in self.padded_captions_idxs_dict[img]:
                X_train.append(images_dir + img + ".npy")
                y_train.append(padded_idx)

        for img in self.test_imgs:
            for padded_idx in self.padded_captions_idxs_dict[img]:
                X_test.append(img)
                y_test.append(padded_idx)

        return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    prepare_images_features()
