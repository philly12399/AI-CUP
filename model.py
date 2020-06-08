import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from loss_function import F_measure_loss
import numpy as np
import pickle
import sys

NAME = 'model_64x3_0dense.model'
tensorboard = TensorBoard(log_dir="logs/" + NAME)

with open("X_train.pickle", "rb") as pickle_in:
	X = pickle.load(pickle_in)
with open("Y_train.pickle", "rb") as pickle_in:
	Y = pickle.load(pickle_in)

model = Sequential()
model.add(Conv2D(64, (5, 2),  input_shape = X.shape[1:]))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Conv2D(64, (4, 2)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(64, (4, 2)))
model.add(Activation('relu'))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=F_measure_loss,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y, batch_size=200, epochs=50, validation_split=0.3, callbacks=[tensorboard])
model.save(NAME) 
