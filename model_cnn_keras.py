from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import datetime
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import matplotlib.pyplot as plt

# Import data
raw_data = np.load('hkflowers.npz')
x_train = raw_data['train_data']
y_train = raw_data['train_labels']
x_test= raw_data['eval_data']
y_test = raw_data['eval_labels']

for i in range(4):
    plt.subplot(221+i)
    plt.imshow((x_train[i]).reshape(50,50,3))

plt.show()
print(y_train[0:4])

from keras.utils import to_categorical

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 50,50,3)
x_test=x_test.reshape(x_test.shape[0],50,50,3)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import regularizers
import keras

#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size=3, padding="same",input_shape=(50, 50, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=3, padding="same"))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=3, padding="same"))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01),))
model.add(Dropout(0.15))
model.add(Dense(3, activation= 'softmax'))

#compile model using accuracy to measure model performance
adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history = model.fit(x_train, y_train, batch_size=3 ,validation_split=0.15, epochs=15)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

e_loss, e_acc=model.evaluate(x=x_test, y=y_test)

print("test set accuracy:", e_acc)





