from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
import pickle
import os

class Colorizer:
    def __init__(self, training_imgs_path):
        self.training_imgs_path = training_imgs_path
        self.training_generator = ImageDataGenerator(rescale=1.0 / 255)
        self.training_set = self.training_generator.flow_from_directory(self.training_imgs_path, target_size=(224, 224),
                                                                        batch_size=64, class_mode=None)
        self.X = []
        self.Y = []

        self.encoder = Sequential()
        self.model = Sequential()

        self.encoder_setup()

    def setup(self, model_path, history_path):
        self.prepare_training_dataset()
        self.model_setup()
        self.fit(200, model_path, history_path)

    def prepare_training_dataset(self):
        for i, batch in enumerate(self.training_set):
            if i > 97:
                break
            print(f"batch: {i} out of {len(self.training_set)}")
            for img in batch:
                L_a_b = rgb2lab(img)
                self.X.append(L_a_b[:, :, 0])
                self.Y.append(L_a_b[:, :, 1:] / 128)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.X = self.X.reshape(self.X.shape + (1,))  # dimensions to be the same for X and Y

    def encoder_setup(self):
        # Encoder
        VGG_model = VGG16()
        for i, layer in enumerate(VGG_model.layers):
            if i < 19:
                self.encoder.add(layer)
        for layer in self.encoder.layers:
            layer.trainable = False

    def model_setup(self):
        # Decoder
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(7, 7, 512)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        self.model.add(UpSampling2D((2, 2)))
        self.model.summary()

    def fit(self, epochs, model_path, history_path):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=15,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        extracted_features = []
        for i, img in enumerate(self.X):
            img = gray2rgb(img)
            img = img.reshape((1, 224, 224, 3))
            prediction = self.encoder.predict(img)
            prediction = prediction.reshape((7, 7, 512))
            extracted_features.append(prediction)

        extracted_features = np.array(extracted_features)

        self.model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
        history = self.model.fit(extracted_features,
                                 self.Y,
                                 verbose=1,
                                 callbacks=[callback],
                                 validation_split=0.2,
                                 epochs=epochs,
                                 batch_size=64)

        self.model.save(model_path)

        try:
            with open(history_path + 'model_history.pkl', 'wb') as f:
                pickle.dump(history, f)
        except:
            pass

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects=None, compile=True)

    def colorize_image(self, image_path, out_path):
        img = img_to_array(load_img(image_path))
        img = resize(img ,(224, 224), anti_aliasing=True)
        img = img * 1.0/255
        lab = rgb2lab(img)
        l = lab[:, :, 0]
        L = gray2rgb(l)
        L = L.reshape((1, 224, 224, 3))
        predicition = self.encoder.predict(L)
        a_b = self.model.predict(predicition)
        a_b = a_b * 128
        out = np.zeros((224, 224, 3))
        out[:, :, 0] = l
        out[:, :, 1:] = a_b
        imsave(out_path + f"{image_path.split('/')[-1][:-4]}.jpg", (lab2rgb(out)*255).astype(np.uint8))

    def colorize_image_directory(self, images_directory, out_directory):
        for filename in os.listdir(images_directory):
            self.colorize_image(f'{images_directory}{filename}', out_directory)


if __name__ == '__main__':
    images_dir = "/home/student/dvir/ML2_Project/Images/"
    colorizer = Colorizer(images_dir)
    colorizer.load_model(path='/home/student/dvir/ML2_Project/colorizing_model/trained_model')
    # colorizer.setup('/home/student/dvir/ML2_Project/colorizing_model/trained_model/', '/home/student/dvir/ML2_Project/colorizing_model/')
    colorizer.colorize_image_directory('/home/student/dvir/ML2_Project/Test_Images/', '/home/student/dvir/ML2_Project/Colorized_Images/Test_Images/')
