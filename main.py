import tensorflow as tf
import csv
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os
import time

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

    return img_to_captions_dict, train_images, test_images


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
        self.caption_padding()

    def captions_preprocess(self):
        for img, caption_list in self.img_to_captions_dict.items():
            processed_captions = []
            for caption in caption_list:
                # allow only word and numbers
                processed_caption = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
                #  allow only words of more than one letter
                processed_caption_list = [word for word in processed_caption.split(' ') if
                                          ((len(word) > 1) and (word.isalpha()))]
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
                padded_idxs = pad_sequences([caption_idxs], maxlen=self.max_caption_len, padding='post')[
                    0]  # todo: maybe value of the padding from zero?
                self.padded_captions_idxs_dict[img].append(padded_idxs)

    def split_dic_to_train_set(self):
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for img in self.train_imgs:
            for padded_idx in self.padded_captions_idxs_dict[img]:
                X_train.append(images_dir + img[:-4] + ".npy")
                y_train.append(padded_idx)

        for img in self.test_imgs:
            for padded_idx in self.padded_captions_idxs_dict[img]:
                X_test.append(images_dir + img[:-4] + ".npy")
                y_test.append(padded_idx)

        return X_train, y_train, X_test, y_test


class AttentionModel(tf.keras.Model):
    def __init__(self, dim):
        super(AttentionModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(dim)
        self.layer2 = tf.keras.layers.Dense(dim)
        self.pred_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, hidden, features):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.layer1(features) + self.layer2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.pred_layer(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer and allows choosing a subset of them
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.layer1 = tf.keras.layers.Dense(embedding_dim)

    @tf.function
    def call(self, x):
        x = self.layer1(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, dim, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.dim = dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # TODO: try with bidirectional
        self.lstm = tf.keras.layers.LSTM(self.dim, return_sequences=True, return_state=True)
        self.layer1 = tf.keras.layers.Dense(self.dim)
        # TODO: check option to ommit fc1
        self.layer2 = tf.keras.layers.Dense(vocab_size)
        self.attention = AttentionModel(self.dim)

    @tf.function
    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the LSTM
        h_seq, state = self.lstm(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.layer1(h_seq)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.layer2(x)

        # TODO: check option to remove attention_weights
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


if __name__ == "__main__":
    # preprocess images and captions and split into train and test sets
    img_to_captions_dict, train_imgs, test_imgs = prepare_images_features()
    prepare_captions = prepare_captions(img_to_captions_dict, train_imgs, test_imgs)
    X_train, y_train, X_test, y_test = prepare_captions.split_dic_to_train_set()

    # create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(lambda x, y: (tf.numpy_function(np.load, x, tf.float32), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # training
    embedding_dim = 256
    units = 512
    vocab_size = prepare_captions.vocab_size
    num_steps = len(X_train) / BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    encoder = Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


    loss_plot = []


    def train_step(img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([prepare_captions.tokenizer.word_index['start_cap']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss



    start_epoch = 0
    EPOCHS = 20

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(train_dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss / int(target.shape[1])
                print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        print(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')


    os.system("find /home/student/dvir/ML2_Project/Flicker8k_Dataset -name '*.npy' -delete")
    print('hi')
