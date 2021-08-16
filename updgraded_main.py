import tensorflow as tf
import csv
from collections import defaultdict
import random
import time
from ImageFeaturesLoader import imagefeaturesloader
from CaptionProcessoer import captionprocessor
import pickle
from FeatureExtractor import featureextractor
from RNNDecoder import rnndecoder
from DenseEnconder import denseenconder
import os

random.seed = 42  # Fixing randomness
images_dir = "/home/student/dvir/ML2_Project/Flicker8k_Dataset/"
tokens_dir = "Flickr8k_text/Flickr8k.token.txt"
image_height = 299
image_width = 299
BATCH_SIZE = 64
BUFFER_SIZE = 1000


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
            img_to_captions_dict[f'{images_dir}{img}'].append(caption)
    return img_to_captions_dict


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([caption_processor.tokenizer.word_index['startcap']] * target.shape[0], 1)

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


if __name__ == "__main__":
    img_to_captions_dict = parse_images_captions_file(tokens_dir)
    # preprocess images and captions and split into train and test sets
    feature_extractor = featureextractor()
    image_features_loader = imagefeaturesloader(list(img_to_captions_dict.keys()), images_dir, feature_extractor)
    train_imgs, test_imgs = image_features_loader.load_images()
    caption_processor = captionprocessor(img_to_captions_dict, train_imgs, test_imgs)
    X_train, y_train, X_test, y_test = caption_processor.split_dic_to_train_set()

    # Creating train dataset by mapping image feature to its caption
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(
        lambda x, y: tf.compat.v1.numpy_function(
            lambda img, cap: (image_features_loader.images_features_dict[img.decode('utf-8')], cap), [x, y],
            [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # training
    embedding_dim = 256
    dims = 512
    vocab_size = caption_processor.vocab_size  # 8425
    num_steps = len(X_train) // BATCH_SIZE
    # Shape feature vectors == (64, 2048)
    features_shape = 2048
    attention_features_shape = 64

    encoder = denseenconder(embedding_dim)
    decoder = rnndecoder(embedding_dim, dims, vocab_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    loss_plot = []

    start_epoch = 0
    EPOCHS = 1

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(train_dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss / int(target.shape[0])
                print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        print(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    # checkpoint_path = "~/dvir/ML2_Project/training/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    # # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    # encoder.save('encoder')
    # decoder.save('decoder')
