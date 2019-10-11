
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load data into train and test sets


# map data to input and output arrays
x = []
y = []
for direct in glob.iglob('data/*/'):
    author = direct.split('\\')[1]
    for filepath in glob.iglob(direct + '*.png'):
        x.append(filepath)
        y.append(author)
x = np.asarray(x)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.167, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_val.shape, y_val.shape)
# plt.imshow(X_train[0])
# plt.show()

# maybe only for Theano
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# print(X_train.shape)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
#
#
# print(y_train.shape)
# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)
# print(y_train.shape)
#
# # Model Architecture: This is where most of the work is
# model = Sequential()
# #relu is Rectified Linear Unit
# model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
# print(model.output_shape)
#
# #more layers
# model.add(Convolution2D(32, 3, 3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25)) #regularizing to prevent overfitting
#
# # add fully connected layer and output layer
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
# print(model.output_shape)
# compile model
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train,
#           batch_size=32, nb_epoch=10, verbose=1)
# Epoch 1/10
# 7744/60000 [==>...........................] - ETA: 96s - loss: 0.5806 - acc: 0.8164

#evaluate model
# score = model.evaluate(X_test, y_test, verbose=0)
