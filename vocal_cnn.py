# -*- coding: utf-8 -*-
"""VOCAL_CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WPiwOGHmyDxtpGKZz9v0NMboBZQu5Nb9
"""

import matplotlib.pyplot as plt
import os
import cv2
from keras.utils import to_categorical

dir_path = '/content/drive/MyDrive/Year 3/Voice AI/Data2/Data_CNN/'
images = []
labels = []
counter = 0
for filename in os.listdir(dir_path):
    if filename.endswith('H.jpg') or filename.endswith('U.jpg'):
      image = cv2.imread(os.path.join(dir_path, filename))
      image = cv2.resize(image, (264, 264))
      images.append(image)
      counter += 1
      print(counter)
      if filename.endswith('U.jpg'):
        # 1 indicates patient is ill
        labels.append(1)
      else:
        # 0 indicates patient is healthy
        labels.append(0)

images = np.asarray(images).astype('float32')
labels = np.asarray(labels).astype('float32')

images /= 255.

print(images.shape)
print(labels.shape)

#Parameters for shuffling
seed = 42
np.random.seed(seed)
N = len(images)

shuffled_indices = np.random.permutation(N)

# Apply the shuffled indices to both data and labels
images = images[shuffled_indices]
labels = labels[shuffled_indices]


split_size_val = int(0.7*len(images))
split_size_test = int(0.85*len(images))

x_train, x_val, x_test = images[:split_size_val], images[split_size_val:split_size_test], images[split_size_test:]
y_train, y_val, y_test = labels[:split_size_val], labels[split_size_val:split_size_test], labels[split_size_test:]

# Print shape of training and val images
print('X_train image shape: {0}'.format(x_train.shape))
print('X_val image shape: {0}'.format(x_val.shape))
print('X_test image shape: {0}'.format(x_test.shape))

# Print shape of training and val labels
print('Y_train labels shape: {0}'.format(y_train.shape))
print('Y_val labels shape: {0}'.format(y_val.shape))
print('Y_test labels shape: {0}'.format(y_test.shape))

import tensorflow as tf
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import legacy
from keras.layers.experimental import preprocessing
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

#data_augmentation = Sequential(
#    [
#        preprocessing.RandomRotation(0.05),
#        preprocessing.RandomFlip("horizontal_and_vertical")
#    ]
#)



model = Sequential()

#Convolutional Layers:


#model.add(data_augmentation)
model.add(Input(shape=(264, 264, 3)))

#64x64 Input
model.add(Conv2D(32, (3, 3), padding='same', strides=2))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
#32x32 Input
model.add(Conv2D(64, (3,3), padding='same', strides=2))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
#16x16 Input
model.add(Conv2D(128, (3,3), padding='same', strides=2))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(BatchNormalization())


#FC Layers:
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Hyper Parameters

opt = legacy.RMSprop(1e-5)

model.summary()

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val,y_val))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])