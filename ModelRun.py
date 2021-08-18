import tensorflow as tf
import csv
from collections import defaultdict
import time
from ImageFeaturesLoader import imagefeaturesloader
from CaptionProcessoer import captionprocessor
from RNNDecoder import rnndecoder
from DenseEnconder import denseenconder
import tensorflow_text as text
import numpy as np
import pickle


class modelrun:
    def __init__(self, params_dict, feature_extractor, load_images=False, optimizer=None, loss_object=None):
        self.images_dir = params_dict['images_dir']
        self.tokens_dir = params_dict['tokens_dir']
        self.batch_size = params_dict['batch_size']
        self.dataset_path = params_dict['dataset_path']
        self.load_images = load_images
        self.embedding_dim = params_dict['embedding_dim']
        self.dims = params_dict['dims']
        # Shape feature vectors == (64, 2048)
        self.features_shape = params_dict['features_shape']
        self.attention_features_shape = params_dict['attention_features_shape']

        self.feature_extractor = feature_extractor
        self.img_to_captions_dict = self.parse_images_captions_file(self.tokens_dir)
        self.caption_processor = captionprocessor(
            self.img_to_captions_dict) if self.dataset_path is not None else None
        self.vocab_size = self.caption_processor.vocab_size if self.dataset_path is not None else None  # 8425

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
        image_features_loader = imagefeaturesloader(list(self.img_to_captions_dict.keys()), self.images_dir,
                                                    self.feature_extractor)
        image_features_loader.load_images()

        # self.caption_processor = captionprocessor(self.img_to_captions_dict, train_imgs, test_imgs)
        self.caption_processor = captionprocessor(self.img_to_captions_dict)
        self.vocab_size = self.caption_processor.vocab_size

        X = []
        y = []

        for img, padded_captions_matrix in self.caption_processor.padded_captions_idxs_dict.items():
            for padded_vec in padded_captions_matrix:
                X.append(img)
                y.append(padded_vec)

        # TODO: remove X_train, y_train, X_test, y_test = self.caption_processor.split_dic_to_train_set()

        # Creating train dataset by mapping image feature to its caption
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.map(
            lambda x, y: tf.compat.v1.numpy_function(
                lambda img, cap: (image_features_loader.images_features_dict[img.decode('utf-8')], cap), [x, y],
                [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        tf.data.experimental.save(dataset, self.dataset_path)

        return dataset

    def train(self, epochs=20, patience=3, min_delta=0.05):
        # Loading saved dataset
        dataset = tf.data.experimental.load(
            self.dataset_path) if not self.load_images else self.prepare_dataset()

        test__val_dataset = dataset.take(2000)
        test_dataset = test__val_dataset.take(1000)
        val_dataset = test__val_dataset.skip(1000)

        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_dataset = dataset.skip(2000)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # Training
        # num_steps = len(X_train) // BATCH_SIZE
        # TODO: check if len(train_dataset) == 6000
        # num_steps = 6000 // self.batch_size

        train_loss_results = []
        train_accuracy_results = []
        validation_avg_f_measure_history = []
        overfit = (0, False)

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()

            start = time.time()
            total_loss = 0
            f_per_batch = []
            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                # Track progress
                epoch_loss_avg.update_state(batch_loss)  # Add current batch loss
                f_per_batch.append(text.metrics.rouge_l(tf.ragged.constant(target.numpy()),
                                                        tf.ragged.constant(self.predict_per_batch(img_tensor,
                                                                                                  target))).f_measure)

                if batch % 100 == 0:
                    average_batch_loss = batch_loss / int(target.shape[0])
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')

            # end epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(np.mean(np.array(f_per_batch)))

            # print(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

            self.encoder.save_weights(f'/home/student/dvir/ML2_Project/encoder_weights/encoder_weight_{epoch}')
            self.decoder.save_weights(f'/home/student/dvir/ML2_Project/decoder_weights/decoder_weights_{epoch}')

            validation_f_per_batch = []
            for (batch, (img_tensor, target)) in enumerate(val_dataset):
                predictions = self.predict_per_batch(img_tensor, target)
                result = text.metrics.rouge_l(tf.ragged.constant(target.numpy()),
                                              tf.ragged.constant(predictions)).f_measure
                avg_f_measure = np.mean(result.f_measure.numpy())
                validation_f_per_batch.append(avg_f_measure)

            validation_avg_f_measure_history.append(np.mean(np.array(validation_f_per_batch)))
            counter = 0
            for i in range(1, len(validation_avg_f_measure_history)):
                if validation_avg_f_measure_history[i] - validation_avg_f_measure_history[i - 1] < min_delta:
                    counter += 1
                else:
                    counter = 0
                if counter >= patience:
                    overfit = (epoch, True)

            if overfit[1]:
                print("overfit")
                break

        metrics_dict = {'train_loss': train_loss_results, 'train_acc': train_accuracy_results,
                        'val_acc': validation_avg_f_measure_history,
                        'final_epoch': overfit[0]}

        with open('metrics_dict.pkl', 'rb') as f:
            pickle.dump(metrics_dict, f)

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

    def predict_per_batch(self, img_tensor, target):
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.caption_processor.tokenizer.word_index['startcap']] * target.shape[0], 1)

        res = np.empty((self.batch_size, 0), dtype=np.int32)
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)
                predicted_id = tf.random.categorical(predictions, 1).numpy()
                res = np.concatenate((res, predicted_id), axis=1)
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        return res

    def perdict_caption(self, image_path=None, features=None):
        hidden = self.decoder.reset_state(batch_size=1)

        dec_input = tf.expand_dims([self.caption_processor.tokenizer.word_index['startcap']], 0)
        caption = []
        caption_vec = []

        if features is None:
            temp_input = tf.expand_dims(self.feature_extractor.load_image(image_path)[0], 0)
            img_tensor_val = self.feature_extractor.image_features_extract_model(temp_input)
            img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                         -1,
                                                         img_tensor_val.shape[3]))
            features = self.encoder(img_tensor_val)

        for i in range(self.caption_processor.max_caption_len):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()  # TODO: check argmax instead

            if self.caption_processor.tokenizer.index_word[predicted_id] == 'endcap':
                return " ".join(caption), caption_vec

            caption.append(self.caption_processor.tokenizer.index_word[predicted_id])
            caption_vec.append(predicted_id)

            dec_input = tf.expand_dims([predicted_id], 0)

        return " ".join(caption), caption_vec

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
