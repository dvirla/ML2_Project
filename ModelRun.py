import tensorflow as tf
import csv
from collections import defaultdict
import time
from ImageFeaturesLoader import imagefeaturesloader
from CaptionProcessor import captionprocessor
from RNNDecoder import rnndecoder
from DenseEnconder import denseenconder
import tensorflow_text as text
import numpy as np
import pickle
import os


class modelrun:
    def __init__(self, params_dict, feature_extractor, load_images=False, optimizer=None, loss_object=None):
        self.images_dirs = params_dict['images_dirs']
        self.tokens_dir = params_dict['tokens_dir']
        self.batch_size = params_dict['batch_size']
        self.dataset_path = params_dict['dataset_path']
        self.load_images = load_images
        self.embedding_dim = params_dict['embedding_dim']
        self.dims = params_dict['dims']
        self.features_shape = params_dict['features_shape']
        self.attention_features_shape = params_dict['attention_features_shape']

        self.feature_extractor = feature_extractor
        self.img_to_captions_dict = self.parse_images_captions_file()
        self.caption_processor = captionprocessor(self.img_to_captions_dict) if self.dataset_path is not None else None
        self.vocab_size = self.caption_processor.vocab_size if self.dataset_path is not None else None

        self.decoder = rnndecoder(self.embedding_dim, self.dims, self.vocab_size)
        self.encoder = denseenconder(self.embedding_dim)

        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam()
        self.loss_object = loss_object if loss_object is not None else tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def parse_images_captions_file(self):
        """
        Reading the image-captions file
        :return: a dictionary {image_name: list of captions}
        """
        img_to_captions_dict = defaultdict(list)
        with open(self.tokens_dir) as f:
            csvreader = csv.reader(f, delimiter='\t')
            for row in csvreader:
                img_caption, caption = row
                img = img_caption.split('#')[0]
                img_to_captions_dict[f'{img}'].append(caption)
        return img_to_captions_dict

    def prepare_dataset(self):
        # preprocess images and captions and split into train and test sets
        image_features_loader = imagefeaturesloader(list(self.img_to_captions_dict.keys()), self.images_dirs,
                                                    self.feature_extractor)
        image_features_loader.load_images()
        self.caption_processor = captionprocessor(self.img_to_captions_dict)
        self.vocab_size = self.caption_processor.vocab_size

        X = []
        y = []

        for img, padded_captions_matrix in self.caption_processor.padded_captions_idxs_dict.items():
            for directory in self.images_dirs:
                directory_images_list = os.listdir(directory)
                if img in directory_images_list:
                    for padded_vec in padded_captions_matrix:
                        X.append(f'{directory}{img}')
                        y.append(padded_vec)

        # Creating train dataset by mapping image feature to its caption
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(
            lambda x, y: tf.compat.v1.numpy_function(
                lambda image, cap: (image_features_loader.images_features_dict[image.decode('utf-8')], cap), [x, y],
                [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        tf.data.experimental.save(dataset, self.dataset_path)

        return dataset

    def train(self, epochs=20):
        # Loading saved dataset or creating new one if there is no saved dataset
        dataset = tf.data.experimental.load(
            self.dataset_path) if not self.load_images else self.prepare_dataset()
        val_dataset = dataset.take(1000)
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_dataset = dataset.skip(1000)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Training
        train_loss_results = []
        train_accuracy_results = []
        validation_avg_f_measure_history = []
        validation_loss_history = []

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()

            start = time.time()
            total_loss = 0
            f_per_batch = []
            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                # Track progress
                epoch_loss_avg.update_state(t_loss)  # Add current batch loss
                # Adding current batch precision
                f_per_batch.append(text.metrics.rouge_l(tf.ragged.constant(target.numpy()),
                                                        tf.ragged.constant(self.predict_per_batch(img_tensor,
                                                                                                  target)[0]),
                                                        alpha=1).f_measure.numpy())

                if batch % 100 == 0:
                    print(f'finished batch number {batch} in epoch {epoch + 1}')
            # end epoch
            train_loss_results.append(epoch_loss_avg.result())
            f_per_batch = np.array(f_per_batch)
            if f_per_batch[-2].shape != f_per_batch[-1].shape:
                train_avg_f_per_epoch = sum([sum(x) for x in f_per_batch]) / (
                            len(f_per_batch[-2]) * (f_per_batch.shape[0] - 1) + len(f_per_batch[-1]))
            else:
                train_avg_f_per_epoch = np.mean(np.array(f_per_batch))
            print(f'Epoch: {epoch + 1}, train average f measure: {train_avg_f_per_epoch}')
            train_accuracy_results.append(train_avg_f_per_epoch)

            print(f'Time taken for epoch {epoch + 1} is {time.time() - start:.2f} sec\n')

            self.encoder.save_weights(f'/home/student/dvir/ML2_Project/encoder_weights_with_grey/encoder_weight_{epoch}.ckpt')
            self.decoder.save_weights(f'/home/student/dvir/ML2_Project/decoder_weights_with_grey/decoder_weights_{epoch}.ckpt')

            # Tracking current validation loss and precision
            validation_f_per_batch = []
            validation_loss_per_batch = []
            for (batch, (img_tensor, target)) in enumerate(val_dataset):
                predictions, t_loss = self.predict_per_batch(img_tensor, target)
                result = text.metrics.rouge_l(tf.ragged.constant(target.numpy()),
                                              tf.ragged.constant(predictions), alpha=1).f_measure.numpy()
                validation_f_per_batch.append(result)
                validation_loss_per_batch.append(t_loss)

            val_avg_f_per_epoch = np.array(validation_f_per_batch)
            validation_loss_per_epoch = np.mean(np.array(validation_loss_per_batch))

            if val_avg_f_per_epoch[-2].shape != val_avg_f_per_epoch[-1].shape:
                val_avg_f_per_epoch = sum([sum(x) for x in val_avg_f_per_epoch]) / (
                            len(val_avg_f_per_epoch[-2]) * (val_avg_f_per_epoch.shape[0] - 1) + len(
                        val_avg_f_per_epoch[-1]))

            else:
                val_avg_f_per_epoch = np.mean(val_avg_f_per_epoch)

            print(f'Epoch: {epoch + 1}, train average f measure: {val_avg_f_per_epoch}')
            validation_avg_f_measure_history.append(val_avg_f_per_epoch)
            validation_loss_history.append(validation_loss_per_epoch)

        metrics_dict = {'train_loss': train_loss_results, 'train_acc': train_accuracy_results,
                        'val_acc': validation_avg_f_measure_history, 'val_loss': validation_loss_history}

        with open('metrics_dict_with_grey.pkl', 'wb') as f:
            pickle.dump(metrics_dict, f)

    def loss_function(self, real, pred):
        """
        :param real: Real captions from train/validation sets
        :param pred: Generated captions for train/validation sets
        :return: Final loss for current generated captions
        """
        # Creating a mask to ignore the zeros from the real captions = 'startcap' word before reducing to sum since
        # we are not interested in this generated word
        mask = tf.math.logical_not(tf.math.equal(real, 0))

        # SparseCategoricalCrossentropy between the real and generated captions
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        """
        :param img_tensor: Batch of images features
        :param target: Batch of the real target captions
        :return: Total loss and average loss over the batch
        """
        loss = 0

        # initializing the hidden state for each batch
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        # Initializing the new captions to 'startcap'
        dec_input = tf.expand_dims([self.caption_processor.tokenizer.word_index['startcap']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder and generating the loss over the predicted captions
                predictions, hidden = self.decoder(dec_input, features, hidden)
                loss += self.loss_function(target[:, i], predictions)
                # "making room" for new word to come
                dec_input = tf.expand_dims(target[:, i], 1)
        avg_loss = (loss / int(target.shape[1]))

        # Backpropogating the gradients for all trainable variables
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, avg_loss

    def predict_per_batch(self, img_tensor, target):
        """
        :param img_tensor: Batch of images features
        :param target: Batch of the real target captions
        :return: New generated captions for the given batch and average loss
        """
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([self.caption_processor.tokenizer.word_index['startcap']] * target.shape[0], 1)

        res = np.empty((target.shape[0], 0), dtype=np.int32)
        loss = 0
        features = self.encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden = self.decoder(dec_input, features, hidden)
            loss += self.loss_function(target[:, i], predictions)
            # Adding new words to current captions based on the new predictions
            predicted_id = tf.random.categorical(predictions, 1).numpy()
            res = np.concatenate((res, predicted_id), axis=1)
            # "making room" for new word to come
            dec_input = tf.expand_dims(target[:, i], 1)
        avg_loss = (loss / int(target.shape[1]))

        return res, avg_loss

    def perdict_caption(self, image_path=None, features=None):
        """
        :param image_path: Path to load image, if None features needs to be provided
        :param features: Image features, if None image path should be provided
        :return: Real caption and its vector representation
        """
        hidden = self.decoder.reset_state(batch_size=1)
        dec_input = tf.expand_dims([self.caption_processor.tokenizer.word_index['startcap']], 0)
        caption = []
        caption_vec = []

        # Loading features if needed
        if features is None:
            temp_input = tf.expand_dims(self.feature_extractor.load_image(image_path)[0], 0)
            img_tensor_val = self.feature_extractor.image_features_extract_model(temp_input)
            img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                         -1,
                                                         img_tensor_val.shape[3]))
            features = self.encoder(img_tensor_val)

        for i in range(self.caption_processor.max_caption_len):
            predictions, hidden = self.decoder(dec_input, features, hidden)
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            # If the new predicted word is 'endcap' stop predicting and return output
            if self.caption_processor.tokenizer.index_word[predicted_id] == 'endcap':
                return " ".join(caption), caption_vec

            caption.append(self.caption_processor.tokenizer.index_word[predicted_id])
            caption_vec.append(predicted_id)
            dec_input = tf.expand_dims([predicted_id], 0)

        return " ".join(caption), caption_vec
