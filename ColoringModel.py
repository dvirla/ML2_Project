from tensorflow.keras.layers import Conv2D, Conv3D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, \
    concatenate
# from keras.layers import Activation, Dense, Dropout, Flatten
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, Model
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
from time import time
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image, ImageFile
import pickle


def rescaling(input_image):
    input_image = input_image / 255.0
    return input_image


class coloringmodel:
    def __init__(self, images_dir, batch_size, img_height, img_width, feature_extractor=None):
        # self.train_datagen = ImageDataGenerator(rescale=1. / 255)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.img_height = img_height
        self.img_width = img_width
        self.train_ds = None
        self.test_ds = None
        self.X = []
        self.Y = []
        self.model = None
        self.create_train_dataset()
        self.define_model()
        self.train()
        self.postprocess()

    def create_train_dataset(self):
        # train = self.train_datagen.flow_from_directory(self.images_dir, target_size=(256, 256),
        #                                                batch_size=self.batch_size,
        #                                                class_mode=None)  # Resizing images to (256, 256)

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.images_dir,
            validation_split=0.2,
            subset="training",
            label_mode=None,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.train_ds.map(rescaling)

        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.images_dir,
            validation_split=0.2,
            subset="validation",
            label_mode=None,
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.test_ds.map(rescaling)

        for batch in self.train_ds:
            for img in batch:
                try:
                    lab = rgb2lab(img)
                    self.X.append(lab[:, :, 0])
                    self.Y.append(lab[:, :, 1:] / 128)  # Dividing by 128 since ab values between -128 and 128
                except:
                    continue
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.X = self.X.reshape(self.X.shape + (1,))  # To fit y shape
        print('X shape: ', self.X.shape)
        print('Y shape is: ', self.Y.shape)

    def define_model(self):
        # Encoder
        encoder_input = Input(shape=(256, 256, 1,))
        encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)

        # Decoder
        decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        self.model = Model(inputs=encoder_input, outputs=decoder_output)

    def train(self, optimizer='adam', loss='mse', metrics=['accuracy'], validation_split=0.2, epochs=1000):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=15,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        history = self.model.fit(self.X,
                                 self.Y,
                                 validation_split=validation_split,
                                 epochs=epochs,
                                 callbacks=[callback],
                                 batch_size=self.batch_size)
        self.model.save('/home/student/dvir/ML2_Project/colorizing_model/model_history.pkl')
        try:
            with open('/home/student/dvir/ML2_Project/colorizing_model/trained_model', 'wb') as f:
                pickle.dump(history, f)
        except:
            pass

    def postprocess(self):
        color_me = []
        for img in self.test_ds:
            # img = img_to_array(load_img(test_path + imgName))
            # img = resize(img, (256, 256))
            color_me.append(img)
        color_me = np.array(color_me, dtype=float)
        color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
        color_me = color_me.reshape(color_me.shape + (1,))
        print('color_me shape: ', color_me.shape)
        output = self.model.predict(color_me)
        output = output * 128

        # Output colorizations
        for i in range(len(output)):
            result = np.zeros((256, 256, 3))
            result[:, :, 0] = color_me[i][:, :, 0]
            result[:, :, 1:] = output[i]
            imsave("/home/student/dvir/ML2_Project/painted_images/" + str(i) + ".png", lab2rgb(result))


if __name__ == '__main__':
    images_dir = "/home/student/dvir/ML2_Project/Flicker8k_Dataset/"
    image_height = 256
    image_width = 256
    batch_size = 64
    coloring_model = coloringmodel(images_dir=images_dir, batch_size=batch_size, img_height=image_height,
                                   img_width=image_width)
