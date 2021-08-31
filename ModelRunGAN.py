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
from ModelRun import modelrun as generator
from Discriminator import discriminator


class modelrungan:
    def __init__(self, params_dict, feature_extractor, load_images=False, optimizer=None, loss_object=None):
        self.generator = generator(params_dict, feature_extractor, load_images, optimizer, loss_object)
        self.discriminator = discriminator(params_dict['embedding_dim'], params_dict['dims'], self.generator.caption_processor.vocab_size)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def train(self, epochs=20):
        dataset = tf.data.experimental.load(
            self.generator.dataset_path) if not self.generator.load_images else self.generator.prepare_dataset()

        val_dataset = dataset.take(1000)

        val_dataset = val_dataset.batch(self.generator.batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_dataset = dataset.skip(1000)
        train_dataset = train_dataset.batch(self.generator.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Training
        train_loss_gen_results = []
        train_loss_disc_results = []
        train_accuracy_results = []
        validation_avg_f_measure_history = []
        validation_loss_history = []

        for epoch in range(epochs):
            epoch_gen_loss_avg = tf.keras.metrics.Mean()
            epoch_disc_loss_avg = tf.keras.metrics.Mean()

            start = time.time()
            total_gen_loss = 0
            total_disc_loss = 0
            f_per_batch = []
            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                gen_loss, disc_loss = self.train_step(img_tensor, target)
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss

                # Track progress
                epoch_gen_loss_avg.update_state(total_gen_loss)  # Add current batch loss
                epoch_disc_loss_avg.update_state(total_disc_loss)
                f_per_batch.append(text.metrics.rouge_l(tf.ragged.constant(target.numpy()),
                                                        tf.ragged.constant(self.generator.predict_per_batch(img_tensor,
                                                                                                            target)[0]),
                                                        alpha=1).f_measure.numpy())
                if batch % 100 == 0:
                    print(f'finished batch number {batch} in epoch {epoch + 1}')
                # end epoch
            train_loss_gen_results.append(epoch_gen_loss_avg.result())
            train_loss_disc_results.append(epoch_disc_loss_avg.result())

            f_per_batch = np.array(f_per_batch)
            if f_per_batch[-2].shape != f_per_batch[-1].shape:
                train_avg_f_per_epoch = sum([sum(x) for x in f_per_batch]) / (
                        len(f_per_batch[-2]) * (f_per_batch.shape[0] - 1) + len(f_per_batch[-1]))
            else:
                train_avg_f_per_epoch = np.mean(np.array(f_per_batch))
            print(f'Epoch: {epoch + 1}, train average f measure: {train_avg_f_per_epoch}')
            train_accuracy_results.append(train_avg_f_per_epoch)

            print(f'Time taken for epoch {epoch + 1} is {time.time() - start:.2f} sec\n')

            self.generator.encoder.save_weights(
                f'/home/student/dvir/ML2_Project/encoder_weights_gan/encoder_weight_{epoch}.ckpt')
            self.generator.decoder.save_weights(
                f'/home/student/dvir/ML2_Project/decoder_weights_gan/decoder_weights_{epoch}.ckpt')

            validation_f_per_batch = []
            validation_loss_per_batch = []
            for (batch, (img_tensor, target)) in enumerate(val_dataset):
                predictions, t_loss = self.generator.predict_per_batch(img_tensor, target)
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

        metrics_dict = {'train_gen_loss': train_loss_gen_results, 'train_loss_disc_results': train_loss_disc_results,
                        'train_acc': train_accuracy_results, 'val_acc': validation_avg_f_measure_history,
                        'val_loss': validation_loss_history}

        with open('metrics_dict_gan.pkl', 'wb') as f:
            pickle.dump(metrics_dict, f)

    @tf.function
    def train_step(self, img_tensor, target):
        hidden = self.generator.decoder.reset_state(batch_size=target.shape[0])

        gen_dec_input = tf.expand_dims(
            [self.generator.caption_processor.tokenizer.word_index['startcap']] * target.shape[0], 1)
        predicted_caption = np.empty((target.shape[0], 0), dtype=np.int32)

        gen_loss = 0
        disc_loss = 0

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_features = self.generator.encoder(img_tensor)
            dis_features = self.discriminator.dense_encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden = self.generator.decoder(gen_dec_input, gen_features, hidden)
                try:
                    predicted_id = tf.random.categorical(predictions, 1).numpy()
                    predicted_caption = np.concatenate((predicted_caption, predicted_id), axis=1)

                    real_output = self.discriminator(target[:, i], dis_features)
                    fake_output = self.discriminator(predicted_caption, dis_features)

                    gen_loss += self.generator_loss(fake_output)
                    disc_loss += self.discriminator_loss(real_output, fake_output)

                    gen_dec_input = tf.expand_dims(target[:, i], 1)
                except AttributeError:
                    print(tf.random.categorical(predictions, 1))

        gen_avg_loss = (gen_loss / int(target.shape[1]))
        disc_avg_los = (disc_loss / int(target.shape[1]))

        gen_trainable_variables = self.generator.encoder.trainable_variables + self.generator.decoder.trainable_variables

        gradients_of_generator = gen_tape.gradient(gen_loss, gen_trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_avg_loss, disc_avg_los

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
