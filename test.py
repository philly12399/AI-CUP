import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pickle
"""
pickle_in = open("X_nor.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("Y_nor.pickle", "rb")
Y = pickle.load(pickle_in)
"""
pickle_in = open("X_train.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("Y_train.pickle", "rb")
Y = pickle.load(pickle_in)
for song in X:
    for feat in song:
        mf = max(feat)
        feat = [0]*2 + [x/mf for x in feat] + [0]*2
        break
    break
X_train=[]
X_temp=[]
ss = 0
for song in X:
    length=len(song[0])
    for i in range(length-4):
        print(ss)
        ss+=1
        X_temp = np.array(song[0][i:i+5])
        for f in range(1,len(song)):
            X_temp = np.append(X_temp, np.array(song[f][i:i+5]))
        X_train = np.append(X_train, X_temp)
X = np.array(X_train)
with open("X_nor.pickle", "wb") as pkfile:
    pickle.dump(X, pkfile)
Y_train=[] 
for frame in Y:
    Y_train=Y_train + frame    
Y = np.array(Y_train)
with open("Y_nor.pickle", "wb") as pkfile:
    pickle.dump(Y, pkfile)
print("end")
"""
print(X.shape)
print(Y.shape)
model = Sequential()
model.add(Conv2D(256, (10, 3),  input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (10, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, Y, batch_size=32, epochs=5, validation_split=0.3)
"""