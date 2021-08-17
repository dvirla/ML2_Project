import tensorflow as tf
import csv
from collections import defaultdict
import random
import time
from ImageFeaturesLoader import imagefeaturesloader
from CaptionProcessoer import captionprocessor
from RNNDecoder import rnndecoder
from DenseEnconder import denseenconder


class modelrun:
    def __init__(self, params_dict, feature_extractor, optimizer=None, loss_object=None):
        self.images_dir = params_dict['images_dir']
        self.tokens_dir = params_dict['tokens_dir']
        self.batch_size = params_dict['batch_size']
        self.train_dataset_path = params_dict['train_dataset_path']
        self.embedding_dim = params_dict['embedding_dim']
        self.dims = params_dict['dims']
        # Shape feature vectors == (64, 2048)
        self.features_shape = params_dict['features_shape']
        self.attention_features_shape = params_dict['attention_features_shape']

        self.feature_extractor = feature_extractor
        self.img_to_captions_dict = self.parse_images_captions_file(self.tokens_dir)
        self.caption_processor = captionprocessor(self.img_to_captions_dict) if self.train_dataset_path is not None else None
        self.vocab_size = self.caption_processor.vocab_size if self.train_dataset_path is not None else None  # 8425

        self.decoder = rnndecoder(self.embedding_dim, self.dims, self.vocab_size)
        self.encoder = denseenconder(self.embedding_dim)

        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam()
        self.loss_object = loss_object if loss_object is not None else tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def parse_images_captions_file(self, path):
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
                img_to_captions_dict[f'{self.images_dir}{img}'].append(caption)
        return img_to_captions_dict

    def prepare_dataset(self):
        # preprocess images and captions and split into train and test sets
        image_features_loader = imagefeaturesloader(list(self.img_to_captions_dict.keys()), self.images_dir, self.feature_extractor)
        train_imgs, test_imgs = image_features_loader.load_images()

        self.caption_processor = captionprocessor(self.img_to_captions_dict, train_imgs, test_imgs)
        self.vocab_size = self.caption_processor.vocab_size

        X_train, y_train, X_test, y_test = self.caption_processor.split_dic_to_train_set()

        # Creating train dataset by mapping image feature to its caption
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(
            lambda x, y: tf.compat.v1.numpy_function(
                lambda img, cap: (image_features_loader.images_features_dict[img.decode('utf-8')], cap), [x, y],
                [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        tf.data.experimental.save(train_dataset, self.train_dataset_path)

        return train_dataset

    def train(self, epochs=20, patience=5, min_delta=1e-3):
        # Loading saved dataset
        train_dataset = tf.data.experimental.load(self.train_dataset_path) if self.train_dataset_path is not None else self.prepare_dataset()

        # Training
        # num_steps = len(X_train) // BATCH_SIZE
        # TODO: check if len(train_dataset) == 6000
        num_steps = 6000 // self.batch_size

        loss_history= []

        for epoch in range(epochs):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss / int(target.shape[0])
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')
            # storing the epoch end loss value to plot later
            # loss_history.append(total_loss)
            # early_stop = self.check_earlystop(loss_history, patience, min_delta)
            print(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

        self.encoder.save_weights('/home/student/dvir/ML2_Project/encoder_weights/encoder_weight')
        self.decoder.save_weights('/home/student/dvir/ML2_Project/decoder_weights/decoder_weights')

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.caption_processor.tokenizer.word_index['startcap']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                loss += self.loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    # @tf.function
    # def check_earlystop(self, loss_history, patience, min_delta):
    #     if len(loss_history) < patience:
    #         return True
    #     counter = 0
    #     for i in range(1, len(loss_history) + 1):
    #         curr_loss = loss_history[i]
    #         last_loss = loss_history[i-1]
    #         if curr_loss - last_loss < min_delta:
    #             counter += 1
    #         else:
    #             counter = 0
    #         if counter >= patience:
    #             return True
