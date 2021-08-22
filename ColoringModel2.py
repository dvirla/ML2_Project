from tensorflow.keras.layers import Conv2D, Conv3D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, \
    concatenate, RepeatVector
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf
from PIL import Image, ImageFile
import pickle
from tensorflow.python.framework import ops



def fix_gpu():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


fix_gpu()
# Get images
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2

# print('loading weights')
# #Load weights
# inception = InceptionResNetV2(weights=None, include_top=True)
# inception.load_weights('/home/student/dvir/ML2_Project/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt')
# inception.graph = tf.compat.v1.get_default_graph()

print('loading images')
X = []
for filename in os.listdir('/home/student/dvir/ML2_Project/Flicker8k_Dataset'):
    X.append(resize(img_to_array(load_img('/home/student/dvir/ML2_Project/Flicker8k_Dataset/'+filename)), (299,299,3)))
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X

# Set up training and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain

print('initialize model')
#Design the neural network
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=15,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

# Finish model
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)


# Generate training data
batch_size = 50
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)


# Train model
history = model.fit_generator(image_a_b_gen(batch_size), callbacks=[callback], steps_per_epoch=10000, epochs=1)
# Test images
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))

model.save('/home/student/dvir/ML2_Project/colorizing_model_2/trained_model')
try:
    with open('/home/student/dvir/ML2_Project/colorizing_model_2/model_history.pkl', 'wb') as f:
        pickle.dump(history, f)
except:
    pass

# Load black and white images
color_me = []
for filename in os.listdir('/home/student/dvir/ML2_Project/colorizing_test_imgs/'):
        color_me.append(img_to_array(load_img('/home/student/dvir/ML2_Project/colorizing_test_imgs/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))# Test model
output = model.predict(color_me)
output = output * 128# Output colorizations
for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = color_me[i][:,:,0]
        cur[:,:,1:] = output[i]
        imsave("/home/student/dvir/ML2_Project/painted_images_2/img_"+str(i)+".png", lab2rgb(cur))


# #Create embedding
# def create_inception_embedding(grayscaled_rgb):
#     grayscaled_rgb_resized = []
#     for i in grayscaled_rgb:
#         i = resize(i, (299, 299, 3), mode='constant')
#         grayscaled_rgb_resized.append(i)
#     grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
#     grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
#     with inception.graph.as_default():
#         embed = inception.predict(grayscaled_rgb_resized)
#     return embed
#
# # Image transformer
# datagen = ImageDataGenerator(
#         shear_range=0.4,
#         zoom_range=0.4,
#         rotation_range=40,
#         horizontal_flip=True)#Generate training data
# batch_size = 20
#
# def image_a_b_gen(batch_size):
#     for batch in datagen.flow(Xtrain, batch_size=batch_size):
#         grayscaled_rgb = gray2rgb(rgb2gray(batch))
#         embed = create_inception_embedding(grayscaled_rgb)
#         lab_batch = rgb2lab(batch)
#         X_batch = lab_batch[:,:,:,0]
#         X_batch = X_batch.reshape(X_batch.shape+(1,))
#         Y_batch = lab_batch[:,:,:,1:] / 128
#         yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

# print('training')
# #Train model
# callback = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     min_delta=0.001,
#     patience=15,
#     verbose=0,
#     mode="auto",
#     baseline=None,
#     restore_best_weights=True,
# )
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# history = model.fit_generator(image_a_b_gen(batch_size), callbacks=[callback], epochs=1000, steps_per_epoch=20)
# model.save('/home/student/dvir/ML2_Project/colorizing_model_2/trained_model')
# try:
#     with open('/home/student/dvir/ML2_Project/colorizing_model_2/model_history.pkl', 'wb') as f:
#         pickle.dump(history, f)
# except:
#     pass
# #Make a prediction on the unseen images
# color_me = []
# for filename in os.listdir('/home/student/dvir/ML2_Project/colorizing_test_imgs'):
#     color_me.append(img_to_array(load_img('/home/student/dvir/ML2_Project/colorizing_test_imgs/'+filename)))
# color_me = np.array(color_me, dtype=float)
# color_me = 1.0/255*color_me
# color_me = gray2rgb(rgb2gray(color_me))
# color_me_embed = create_inception_embedding(color_me)
# color_me = rgb2lab(color_me)[:,:,:,0]
# color_me = color_me.reshape(color_me.shape+(1,))# Test model
# output = model.predict([color_me, color_me_embed])
# output = output * 128
# # Output colorizations
# for i in range(len(output)):
#     cur = np.zeros((256, 256, 3))
#     cur[:,:,0] = color_me[i][:,:,0]
#     cur[:,:,1:] = output[i]
#     imsave("/home/student/dvir/ML2_Project/painted_images_2/img_"+str(i)+".png", lab2rgb(cur))
