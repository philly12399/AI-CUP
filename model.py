import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from loss_function import F_measure_loss
import numpy as np
import os
import datetime
import pickle
import sys

NAME = 'r.model'
tensorboard = TensorBoard(log_dir="logs/" + NAME)

pickle_in = open("X_train.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("Y_train.pickle", "rb")
Y = pickle.load(pickle_in)

model = Sequential()
model.add(Conv2D(64, (5, 3),  input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y, batch_size=32, epochs=50, validation_split=0.3, callbacks=[tensorboard])
model.save('model_n.model')
