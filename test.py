import numpy as np
import random
import keras
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
X_train = X_train.reshape(X_train.shape[0], -1) # 等价于X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(X_test.shape[0], -1)    # 等价于X_test = X_test.reshape(10000,784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy',
             optimizer = 'rmsprop', #RMSprop()
             metrics=['accuracy'])
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# metrics means you want to get more results during the training process
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=128,
                   verbose = 1, validation_data=[X_test, y_test])
history.params
score = model.evaluate(X_test, y_test, verbose = 0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
X_test_0 = X_test[0,:].reshape(1,784)
y_test_0 = y_test[0,:]
plt.imshow(X_test_0.reshape([28,28]))
pred = model.predict(X_test_0[:])
print('Label of testing sample', np.argmax(y_test_0))
print('Output of the softmax layer', pred[0])
print('Network prediction:', np.argmax([pred[0]]))